import google.generativeai as genai

# test_voice_conversation.py から完全移植した予約情報
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
        # ★★★ 修正: gemini-2.0-flash に統一 ★★★
        self.model = genai.GenerativeModel('gemini-2.0-flash') 
        self.history = []

    def select_smart_acknowledgment(self, user_text):
        """
        ★★★ test_voice_conversation.py から完全移植 ★★★
        店員の発言内容に応じて適切な相槌を選択

        Args:
            user_text: 店員の発言テキスト

        Returns:
            tuple[str, str]: (音声テキスト, ログ表示用テキスト)
        """
        utterance_lower = user_text.strip()

        # 質問形式（明確な疑問文）
        if any(kw in utterance_lower for kw in ['ございますか', 'でしょうか', 'いかがですか']):
            return "確認しますので、少々お待ち下さい。", "質問形式 → 「確認しますので、少々お待ち下さい。」"

        # 待機要求（店員が作業する）
        if any(kw in utterance_lower for kw in ['お待ちください', '確認します', '代わります', '変わります']):
            return "承知いたしました。", "待機要求 → 「承知いたしました。」"

        # 確認・復唱（店員が情報を確認）
        if '復唱' in utterance_lower or 'かしこまりました' in utterance_lower:
            return "はい。", "確認 → 「はい。」"

        # デフォルト: シンプルな「はい」
        return "はい。", "デフォルト → 「はい。」"

    def process_conversation(self, user_text, history):
        """
        ★★★ test_voice_conversation.py の get_gemini_response を完全移植 ★★★

        Args:
            user_text: 店員の発言
            history: 会話履歴

        Returns:
            tuple[str, list]: (AI応答テキスト, 更新された履歴)
        """
        # 履歴のフォーマット
        history_text = "\n".join([f"{h['role']}: {h['text']}" for h in history[-10:]])

        # ★★★ test_voice_conversation.py から完全移植したプロンプト ★★★
        prompt = f"""あなたはレストラン予約の電話をかけている予約代行AIです。
以下の予約情報で予約を取ってください。丁寧な日本語で簡潔に話してください（1-2文）。

【予約情報】
- 予約者名: {RESERVATION_INFO['reserver_name']}
- 連絡先: {RESERVATION_INFO['contact_phone']}
- 希望日: {RESERVATION_INFO['date']}
- 希望時間: {RESERVATION_INFO['time']}
- 人数: {RESERVATION_INFO['guests']}名
- 席種: {RESERVATION_INFO['seat_type']}
- 時間の融通: {RESERVATION_INFO['flexibility']} （※店から聞かれた場合のみ伝える）
- 備考: {RESERVATION_INFO['notes']}

【重要な指示】
- 電話番号を伝える際は、必ず1桁ずつ区切って伝えてください。
  例: 「090-1234-5678」→「ゼロキュウゼロ、イチニーサンヨン、ゴーロクナナハチ」
- 「6千7百」や「8じゅう9」のような表現は絶対に使わないでください。
- 時間の融通（30分程度前後可能）は、店から「その時間は難しい」などと聞かれた場合のみ伝えてください。聞かれていないのに自分から言わないでください。
- 「当日はよろしくお願いいたします」は使わないでください（これは店側が客に言うセリフです）。
- 店員が「お待ちしております」「ご来店をお待ちしております」などと予約確定を告げた場合は、「ありがとうございました。それでは失礼いたします。」などで締めくくってください。

【絶対に禁止】
- ト書きや括弧書きの説明は絶対に含めないでください。
  ❌ 悪い例: 「（保留音の後、店員に代わる）」「（少々お待ちください）」
  ✅ 良い例: 「承知いたしました。」「はい、お待ちしております。」
- 音声で読み上げられない記号や説明文は一切書かないでください。
- 実際に声に出して話す内容のみを出力してください。

【これまでの会話】
{history_text}

【店員の発言】
{user_text}

【あなたの応答】:"""

        try:
            response = self.model.generate_content(prompt)
            ai_text = response.text.strip()

            # ★★★ Geminiが "AI: " を出力することがあるため除去 ★★★
            if ai_text.startswith("AI: ") or ai_text.startswith("AI:"):
                ai_text = ai_text.replace("AI: ", "", 1).replace("AI:", "", 1).strip()

            # 履歴更新
            new_history = history.copy()
            new_history.append({"role": "店員", "text": user_text})
            new_history.append({"role": "AI", "text": ai_text})

            return ai_text, new_history

        except Exception as e:
            print(f"[Gemini Error] {e}")
            return "少々お待ちください。", history
