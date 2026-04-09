"""FastAPI WebSocket サーバー (Gemini Live API 版)。

旧版では Google Cloud STT (streaming) で音声認識 → Gemini REST で応答生成
→ Google Cloud TTS で音声合成、というパイプラインを自前で組み、さらに
"プリフェッチ第一声" や "相槌 (はい)" などで応答までの間を埋めていた。

本版では Gemini Live API (`gemini-2.5-flash-native-audio-preview-12-2025`)
を使うことで、音声入力 → 音声出力までを単一セッションでネイティブ処理する。
これにより以下の独自ロジックは全て廃止された:

- Google Cloud STT / TTS クライアント
- プリフェッチ第一声生成 (intro_task)
- 相槌生成 (select_smart_acknowledgment)
- エコー除去 (is_semantic_echo)
- SSML タグの除去/重複カット

ブラウザ <-> 本サーバー間は引き続き WebSocket。音声フォーマットは:
- アップリンク: 16 kHz / 16-bit PCM mono (バイナリフレーム)
- ダウンリンク: 24 kHz / 16-bit PCM mono (base64 で JSON 送信)
"""

import asyncio
import base64
import json
import logging
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

from core_logic import ReservationAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# --- 設定 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY が設定されていません")

LIVE_API_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# AI エンジン (RESERVATION_INFO とシステム指示を提供)
ai_engine = ReservationAI(GEMINI_API_KEY)

# Live API クライアント
live_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
async def index() -> HTMLResponse:
    with open("templates/phone.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ WebSocket 接続開始")

    if live_client is None:
        await websocket.send_json({"type": "error", "message": "GEMINI_API_KEY 未設定"})
        await websocket.close()
        return

    # ブラウザから受信した音声バイト列を Live API へ橋渡しするためのキュー
    audio_in_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    is_running = True

    async def pump_browser_to_queue() -> None:
        """ブラウザからの WebSocket メッセージを読み取り、音声はキューに、
        コマンド (interrupt 等) は state 操作に振り分ける。"""
        nonlocal is_running
        try:
            while is_running:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    await audio_in_queue.put(message["bytes"])
                elif "text" in message and message["text"] is not None:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    event = data.get("event")
                    if event == "stop":
                        logger.info("🛑 ブラウザから停止要求")
                        is_running = False
                        break
        except WebSocketDisconnect:
            logger.info("👋 WebSocket 切断")
        except Exception as e:
            logger.error(f"❌ ブラウザ受信エラー: {e}")
        finally:
            is_running = False
            await audio_in_queue.put(None)

    # Live API のセッション設定
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": ai_engine.build_system_instruction(),
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {
                "disabled": False,
                "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
                "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
                "prefix_padding_ms": 100,
                "silence_duration_ms": 500,
            }
        },
    }

    try:
        async with live_client.aio.live.connect(model=LIVE_API_MODEL, config=config) as session:
            logger.info("✅ Live API セッション開始")

            # 通話開始時、AI 側から先に挨拶させるためのトリガー。
            # ダミーのユーザーターンを送ることで model に発話を促す。
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text="（通話が接続されました。挨拶から始めてください）")],
                ),
                turn_complete=True,
            )

            async def forward_audio_to_gemini() -> None:
                """ブラウザ由来の PCM をリアルタイム入力として Live API に送る。"""
                while is_running:
                    try:
                        chunk = await asyncio.wait_for(audio_in_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        return
                    try:
                        await session.send_realtime_input(
                            audio={"data": chunk, "mime_type": "audio/pcm;rate=16000"}
                        )
                    except Exception as e:
                        logger.error(f"❌ Live API 送信エラー: {e}")
                        return

            async def forward_gemini_to_browser() -> None:
                """Live API からの応答をブラウザへ転送する。
                音声は base64、文字起こしと制御イベントは JSON で送る。"""
                while is_running:
                    try:
                        turn = session.receive()
                        async for response in turn:
                            sc = getattr(response, "server_content", None)
                            if sc is None:
                                continue

                            # 割り込み (ユーザーが AI の発話中に話し始めた)
                            if getattr(sc, "interrupted", False):
                                await _safe_send_json(websocket, {"type": "interrupted"})

                            # 入力 (店員) の文字起こし
                            input_tx = getattr(sc, "input_transcription", None)
                            if input_tx is not None and getattr(input_tx, "text", None):
                                await _safe_send_json(
                                    websocket,
                                    {"type": "user_transcript", "text": input_tx.text},
                                )

                            # 出力 (AI) の文字起こし
                            output_tx = getattr(sc, "output_transcription", None)
                            if output_tx is not None and getattr(output_tx, "text", None):
                                await _safe_send_json(
                                    websocket,
                                    {"type": "ai_transcript", "text": output_tx.text},
                                )

                            # AI 音声チャンク
                            model_turn = getattr(sc, "model_turn", None)
                            if model_turn is not None:
                                for part in model_turn.parts:
                                    inline = getattr(part, "inline_data", None)
                                    if inline is not None and inline.data:
                                        audio_b64 = base64.b64encode(inline.data).decode("utf-8")
                                        await _safe_send_json(
                                            websocket,
                                            {"type": "audio", "audio": audio_b64},
                                        )

                            # ターン完了
                            if getattr(sc, "turn_complete", False):
                                await _safe_send_json(websocket, {"type": "turn_complete"})
                    except Exception as e:
                        logger.error(f"❌ Live API 受信エラー: {e}")
                        return

            # ブラウザ受信 / Live API 送信 / Live API 受信 を並行実行
            async with asyncio.TaskGroup() as tg:
                tg.create_task(pump_browser_to_queue(), name="browser_in")
                tg.create_task(forward_audio_to_gemini(), name="gemini_send")
                tg.create_task(forward_gemini_to_browser(), name="gemini_recv")

    except* Exception as eg:
        for e in eg.exceptions:
            logger.error(f"❌ セッションエラー: {type(e).__name__}: {e}")
    finally:
        is_running = False
        logger.info("セッション終了")
        try:
            await websocket.close()
        except Exception:
            pass


async def _safe_send_json(websocket: WebSocket, data: dict) -> None:
    try:
        await websocket.send_json(data)
    except Exception:
        # 切断済み等は無視
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
