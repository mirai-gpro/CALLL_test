import os
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from core_logic import ReservationAI

# Google公式ライブラリ
from google.oauth2 import service_account
from google.cloud import texttospeech

app = FastAPI()

# 鍵の読み込み
gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    print("警告: GEMINI_API_KEY が設定されていません")

# AIの初期化
ai_engine = ReservationAI(gemini_key)

# TTSクライアントの初期化 (google.json 使用)
tts_client = None
try:
    if os.path.exists("google.json"):
        creds = service_account.Credentials.from_service_account_file("google.json")
        tts_client = texttospeech.TextToSpeechClient(credentials=creds)
        print("✅ Google Cloud TTS (google.json) 接続成功")
    else:
        print("❌ エラー: google.json が見つかりません")
except Exception as e:
    print(f"❌ TTS認証エラー: {e}")

def synthesize_speech(text):
    """Google Cloud TTSで音声を生成"""
    if not tts_client: 
        return None

    # ★★★ test_voice_conversation.py と同じ設定 ★★★
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ja-JP",
        name="ja-JP-Chirp3-HD-Leda" 
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0
    )

    try:
        response = tts_client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        return base64.b64encode(response.audio_content).decode('utf-8')
    except Exception as e:
        print(f"[TTS API Error] {e}")
        return None

@app.get("/")
async def get():
    """HTMLページを返す"""
    with open("templates/phone.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続処理"""
    await websocket.accept()
    history = []

    # ★★★ 割り込み制御用のフラグ ★★★
    current_ai_speaking = False  # AI発話中かどうか

    try:
        while True:
            data = await websocket.receive_json()

            # ==========================================
            # 割り込み検知 (Barge-in)
            # ==========================================
            if data.get("event") == "interrupt":
                print("\n[割り込み検知] AI音声を停止")
                current_ai_speaking = False

                # ★★★ クライアントに停止命令を送信 ★★★
                await websocket.send_json({
                    "type": "stop_audio",
                    "message": "AI音声を停止しました"
                })
                continue

            # ==========================================
            # ユーザー発話の処理
            # ==========================================
            user_text = data.get("text")
            if not user_text:
                continue

            print(f"\n[店員] {user_text}")

            # ==========================================
            # 1. 即答相槌 (Smart Acknowledgment)
            # ==========================================
            # ★★★ tuple で受け取る（test_voice_conversation.py と同じ） ★★★
            ack_text, ack_log = ai_engine.select_smart_acknowledgment(user_text)
            print(f"[即答相槌] {ack_log}")

            ack_audio = synthesize_speech(ack_text)
            if ack_audio:
                current_ai_speaking = True
                await websocket.send_json({
                    "type": "audio",
                    "text": ack_text,
                    "audio": ack_audio,
                    "is_acknowledgment": True  # 相槌フラグ
                })

                # ★★★ 相槌の再生時間を待つ（約0.5-1秒） ★★★
                import asyncio
                await asyncio.sleep(0.8)

            # ==========================================
            # 2. Gemini応答生成
            # ==========================================
            print("[Gemini] 応答生成中...")
            ai_text, history = ai_engine.process_conversation(user_text, history)
            print(f"[AI] {ai_text}")

            # ==========================================
            # 3. Gemini応答の音声化と送信
            # ==========================================
            ai_audio = synthesize_speech(ai_text)
            if ai_audio:
                current_ai_speaking = True
                await websocket.send_json({
                    "type": "audio",
                    "text": ai_text,
                    "audio": ai_audio,
                    "is_acknowledgment": False  # Gemini応答
                })
            else:
                await websocket.send_json({
                    "type": "error", 
                    "text": "音声生成エラー"
                })

            current_ai_speaking = False

    except WebSocketDisconnect:
        print("\n[切断] WebSocket接続が閉じられました")
    except Exception as e:
        print(f"\n[エラー] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # ★★★ ポート5000で起動 ★★★
    uvicorn.run(app, host="0.0.0.0", port=5000)
