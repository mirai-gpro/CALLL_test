import os
import base64
import json
import asyncio
import queue
import threading
import time
import concurrent.futures
from difflib import SequenceMatcher
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from core_logic import ReservationAI
import re 

# Google Cloud ライブラリ
from google.oauth2 import service_account
from google.cloud import texttospeech
from google.cloud import speech

app = FastAPI()

# --- 設定 ---
gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    print("⚠️ 警告: GEMINI_API_KEY が設定されていません")

# AIエンジン初期化
ai_engine = ReservationAI(gemini_key)

# 認証情報の読み込み
CREDENTIALS_FILE = "google.json"
creds = None
if os.path.exists(CREDENTIALS_FILE):
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    print("✅ Google Cloud 認証成功 (google.json)")
else:
    print("❌ エラー: google.json が見つかりません")

# クライアント初期化
tts_client = texttospeech.TextToSpeechClient(credentials=creds) if creds else None
stt_client = speech.SpeechClient(credentials=creds) if creds else None

# STT設定 (STT認識強化適用済み - STTコンテキスト維持)
STT_CONFIG = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="ja-JP",
    enable_automatic_punctuation=True,

    # Speech Context (Recognition Context) - 電話番号認識強化
    speech_contexts=[
        speech.SpeechContext(
            phrases=['090', '1234', '5678', 'ゼロキュウゼロ', 'イチニーサンヨン', 'ゴーロクナナハチ'],
            boost=20.0
        ),
        speech.SpeechContext(
            phrases=['電話番号', '090-1234-5678', '09012345678'],
            boost=10.0
        ),
        # 数字列の認識を安定させるためのヒント
        speech.SpeechContext(
            phrases=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'です'],
            boost=5.0
        )
    ]
)
STREAMING_CONFIG = speech.StreamingRecognitionConfig(
    config=STT_CONFIG,
    interim_results=True,
    single_utterance=False
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
connection_states = {}

# 履歴からSSMLタグを削除する関数
def clean_ssml_from_history(history: list):
    clean_history = []
    for entry in history:
        # <prosody...>タグやその他のXMLタグを全て除去
        clean_text = re.sub(r'<[^>]+>', '', entry['text']) 
        clean_history.append({"role": entry['role'], "text": clean_text})
    return clean_history

def synthesize_speech_sync(text):
    if not tts_client: return None
    try:
        # SSMLタグが含まれている場合は、タイプをSSMLに設定
        synthesis_input = texttospeech.SynthesisInput(text=text)
        if '<prosody' in text or '<speak>' in text:
             synthesis_input = texttospeech.SynthesisInput(ssml=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name="ja-JP-Chirp3-HD-Leda"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return base64.b64encode(response.audio_content).decode('utf-8')
    except Exception as e:
        print(f"❌ TTSエラー: {e}")
        return None

async def synthesize_speech(text):
    """同期的なTTS関数を非同期スレッドで実行"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, synthesize_speech_sync, text)

async def process_conversation_async(user_text, history):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, ai_engine.process_conversation, user_text, history)

async def select_smart_ack_async(user_text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, ai_engine.select_smart_acknowledgment, user_text)

async def send_json_safe(websocket: WebSocket, data: dict):
    try:
        await websocket.send_json(data)
    except Exception:
        pass

# エコー判定
def is_semantic_echo(transcript, ai_text):
    if not ai_text or not transcript: return False
    def normalize(text):
        # SSMLタグを除去してから比較
        text = re.sub(r'<[^>]+>', '', text)
        return text.replace(" ", "").replace("　", "").replace("。", "").replace("、", "").strip()

    t_norm = normalize(transcript)
    a_norm = normalize(ai_text)

    # 1. 完全一致
    if t_norm == a_norm: return True

    # 2. 包含 (短い言葉は先頭/末尾一致のみ)
    if t_norm in a_norm:
        if len(t_norm) <= 2:
            return a_norm.startswith(t_norm) or a_norm.endswith(t_norm)
        return True 

    # 3. 類似度（閾値 0.8 に設定）
    ratio = SequenceMatcher(None, t_norm, a_norm).ratio()
    if ratio > 0.90: return True 

    return False

@app.get("/")
async def get():
    with open("templates/phone.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket接続開始")

    # ==========================================================
    # ★プリフェッチ開始
    # ==========================================================
    reserver_name = ai_engine.RESERVATION_INFO['reserver_name']
    default_intro = f"お忙しいところ恐れ入ります。私、{reserver_name}のAIアシスタントです。予約をお願いできますでしょうか。"

    intro_text = getattr(ai_engine, "INTRO_TEXT", default_intro)

    print(f"🚀 [Prefetch] 第一声の生成を開始: {intro_text[:15]}...")
    intro_task = asyncio.create_task(synthesize_speech(intro_text)) 

    audio_queue = queue.Queue()
    loop = asyncio.get_event_loop()
    is_connected = True

    # 状態管理（historyを含めて永続化）
    state = {
        "is_first_interaction": True,
        "current_ai_text": "",
        "intro_task": intro_task,  
        "history": []              
    }

    connection_states[websocket] = state

    def run_stt_loop():
        nonlocal is_connected
        def request_generator():
            while is_connected:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    if chunk is None: return
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    yield speech.StreamingRecognizeRequest(audio_content=b'\x00' * 3200)

        while is_connected:
            try:
                responses = stt_client.streaming_recognize(STREAMING_CONFIG, request_generator())
                for response in responses:
                    if not is_connected: break
                    if not response.results: continue
                    result = response.results[0]
                    if not result.alternatives: continue

                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final

                    current_ai_text = state["current_ai_text"]
                    if is_semantic_echo(transcript, current_ai_text):
                        if is_final: print(f"🔇 エコー除去: '{transcript}'")
                        continue

                    asyncio.run_coroutine_threadsafe(
                        send_json_safe(websocket, {"type": "transcript", "text": transcript, "is_final": is_final}),
                        loop
                    )

                    if is_final:
                        print(f"🗣️ 認識確定: {transcript}")
                        state["current_ai_text"] = ""
                        # 会話処理へ
                        asyncio.run_coroutine_threadsafe(
                            handle_conversation_flow(websocket, transcript, state),
                            loop
                        )
            except Exception as e:
                if is_connected: time.sleep(1)
                else: break

    stt_thread = threading.Thread(target=run_stt_loop, daemon=True)
    stt_thread.start()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_queue.put(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("event") == "interrupt":
                    print("🛑 割り込み受信")
                    state["current_ai_text"] = ""
    except WebSocketDisconnect:
        print("👋 WebSocket切断")
    except Exception as e:
        print(f"❌ WebSocketエラー: {e}")
    finally:
        is_connected = False
        audio_queue.put(None)
        # タスクが完了していなければキャンセル
        if not state["intro_task"].done():
            state["intro_task"].cancel()
        if websocket in connection_states:
            del connection_states[websocket]

async def handle_conversation_flow(websocket: WebSocket, user_text: str, state: dict):
    if not user_text.strip(): return

    # 現在の履歴を取得（stateから参照）
    history = state["history"]

    # ============================================
    # ★ 1. 初回: プリフェッチ済み音声を即回収
    # ============================================
    if state["is_first_interaction"]:
        print("[初回] 店員の第一声を検知 → プリフェッチ音声を回収")
        state["is_first_interaction"] = False

        reserver_name = ai_engine.RESERVATION_INFO['reserver_name']
        default_intro = f"お忙しいところ恐れ入ります。私、{reserver_name}のAIアシスタントです。予約をお願いできますでしょうか。"

        greeting_text = getattr(ai_engine, "INTRO_TEXT", default_intro)

        # エコー判定用にSSMLタグを除去したテキストを保存
        state["current_ai_text"] = re.sub(r'<[^>]+>', '', greeting_text)

        # 履歴に追加
        history.append({"role": "user", "text": user_text})
        history.append({"role": "ai", "text": greeting_text})

        # NOTE: プリフェッチ音声の回収は、タイムアウトなしで完了を待つ
        try:
            ai_audio = await state["intro_task"]

            if ai_audio:
                print(f"🚀 [即答] 第一声を送信: {greeting_text[:10]}...")
                await send_json_safe(websocket, {"type": "audio", "text": greeting_text, "audio": ai_audio})
            else:
                print("❌ 第一声の生成に失敗しました")
        except Exception as e:
            print(f"❌ タスク待機エラー: {e}")

        return # Geminiは呼ばない

    # ============================================
    # ★ 2. 通常: 相槌 + Gemini
    # ============================================

    # --- 相槌 ---
    ack_text = ""
    try:
        ack_text, _ = await select_smart_ack_async(user_text)

        # エコー判定用にSSMLタグを除去したテキストを保存
        state["current_ai_text"] = re.sub(r'<[^>]+>', '', ack_text)

        ack_audio = await synthesize_speech(ack_text)

        if ack_audio:
            await send_json_safe(websocket, {"type": "audio", "text": ack_text, "audio": ack_audio})

            # 固定ウェイト 0.5秒
            await asyncio.sleep(0.5) 

            # デバッグログはSSML除去済みテキストを使用
            clean_ack_text = re.sub(r'<[^>]+>', '', ack_text)
            print(f"✅ 即答送信完了 (Wait: 0.5s): '{clean_ack_text}'")

    except Exception as e:
        print(f"❌ 相槌エラー: {e}")

    # --- Gemini応答 ---
    try:
        # LLMに渡す前に履歴からSSMLタグを削除する
        clean_history = clean_ssml_from_history(history)

        # Gemini処理（戻り値の new_history を受け取る）
        ai_text, new_history = await process_conversation_async(user_text, clean_history)

        # 新しい履歴を state に保存して永続化する
        state["history"] = new_history 

        # 重複カット
        clean_ack_text = re.sub(r'<[^>]+>', '', ack_text)

        if clean_ack_text and ai_text.startswith(clean_ack_text):
            ai_text = ai_text[len(clean_ack_text):].strip()
            if ai_text.startswith("。") or ai_text.startswith("、"):
                ai_text = ai_text[1:].strip()

        print(f"🤖 AI(本題): {ai_text}")

        # 【修正箇所】空文字列や句読点フィルタリングを削除し、LLMの応答をそのまま使用する
        if not ai_text: 
            # LLMが空文字列を返した場合は、そのまま無言で終了する（沈黙）
            return

        state["current_ai_text"] = ai_text
        ai_audio = await synthesize_speech(ai_text)
        if ai_audio:
            await send_json_safe(websocket, {"type": "audio", "text": ai_text, "audio": ai_audio})

            # LLM応答（本題）の後もウェイトを入れる
            await asyncio.sleep(2.0) 

    except Exception as e:
        print(f"❌ AIエラー: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
