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
    if not tts_client: return None

    # ローカルファイル(test_voice_conversation.py)と同じ設定
    # ja-JP-Chirp3-HD-Leda を使用
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
        print(f"TTS API Error: {e}")
        return None

@app.get("/")
async def get():
    with open("templates/phone.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    history = []

    try:
        while True:
            data = await websocket.receive_json()

            # --- 割り込み検知 (Barge-in) ---
            # phone.html から "interrupt" イベントが来たら、
            # 現在の処理をスキップして次の入力を待つ（擬似的な停止）
            if data.get("event") == "interrupt":
                print("--- 割り込み検知: AI処理停止 ---")
                continue

            user_text = data.get("text")
            if user_text:
                print(f"\n店員: {user_text}")

                # ==========================================
                # 1. 即答相槌 (Smart Acknowledgment) の処理
                # ==========================================
                # Geminiを待たずに、まずは相槌を返す (test_voice_conversation.py の挙動)
                ack_text = ai_engine.select_smart_acknowledgment(user_text)
                print(f"[即答] {ack_text}")

                ack_audio = synthesize_speech(ack_text)
                if ack_audio:
                    # 相槌を送信
                    await websocket.send_json({
                        "type": "audio",
                        "text": ack_text,
                        "audio": ack_audio
                    })

                # ==========================================
                # 2. Gemini応答生成
                # ==========================================
                ai_text, history = ai_engine.process_conversation(user_text, history)
                print(f"AI: {ai_text}")

                # ==========================================
                # 3. Gemini応答の音声化と送信
                # ==========================================
                ai_audio = synthesize_speech(ai_text)
                if ai_audio:
                    await websocket.send_json({
                        "type": "audio",
                        "text": ai_text,
                        "audio": ai_audio
                    })
                else:
                    await websocket.send_json({"type": "error", "text": "音声生成エラー"})

    except WebSocketDisconnect:
        print("切断されました")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)