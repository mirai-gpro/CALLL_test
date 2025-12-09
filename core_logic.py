import google.generativeai as genai
import os

# 予約情報
RESERVATION_INFO = {
    "restaurant_name": "リストランテ鈴木",
    "reserver_name": "山田太郎",
    "contact_phone": "090-1234-5678",
    "date": "12月25日",
    "day_of_week": "日曜日",
    "time": "19時",
    "guests": 4,
    "seat_type": "テーブル席",
    "flexibility": "30分程度なら前後可能",
    "notes": "誕生日のお祝い"
}

class ReservationAI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.history = []

    def process_conversation(self, user_text, history):
        history_text = "\n".join([f"{h['role']}: {h['text']}" for h in history[-10:]])
        prompt = f"""
あなたはレストラン予約の電話をかけている予約代行AIです。
以下の予約情報で予約を取ってください。丁寧な日本語で簡潔に話してください（1-2文）。

【予約情報】
- 予約者名: {RESERVATION_INFO['reserver_name']}
- 連絡先: {RESERVATION_INFO['contact_phone']}
- 希望日: {RESERVATION_INFO['date']}
- 希望時間: {RESERVATION_INFO['time']}
- 人数: {RESERVATION_INFO['guests']}名
- 席種: {RESERVATION_INFO['seat_type']}
- 時間の融通: {RESERVATION_INFO['flexibility']}
- 備考: {RESERVATION_INFO['notes']}

【指示】
- 電話番号は「ゼロキュウゼロ」と読み上げやすく伝えてください。
- 実際に声に出して話す内容のみを出力してください（ト書き禁止）。
- 店員が「お待ちしております」と言ったら、「ありがとうございました。失礼いたします。」で会話を終了してください。

【会話履歴】
{history_text}

【店員の発言】
{user_text}

【あなたの応答】:"""

        try:
            response = self.model.generate_content(prompt)
            ai_text = response.text.strip().replace("AI: ", "")
            new_history = history.copy()
            new_history.append({"role": "店員", "text": user_text})
            new_history.append({"role": "AI", "text": ai_text})
            return ai_text, new_history
        except Exception as e:
            print(f"Error: {e}")
            return "申し訳ありません、もう一度お願いします。", history