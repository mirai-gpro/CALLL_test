import google.generativeai as genai
import random
import re
import time
from google.api_core.exceptions import ResourceExhausted


class ReservationAI:

    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        # 予約情報
        self.RESERVATION_INFO = {
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

    def select_smart_acknowledgment(self, user_text: str):
        utterance_lower = user_text.strip()
        ssml_hai_tags = "rate='1.05' pitch='+3st' volume='+3dB'"
        HIGH_TENSION_HAI_VARIATIONS = [
            f"<speak><prosody {ssml_hai_tags}>はい</prosody></speak>",
            f"<speak><prosody {ssml_hai_tags}>はい、</prosody></speak>",
            f"<speak><prosody {ssml_hai_tags}>はい！</prosody></speak>",
            f"<speak><prosody rate='1.10' pitch='+4st'>はい</prosody></speak>",
        ]

        if any(kw in utterance_lower
               for kw in ['待って', '待ち', '確認し', '代わり', '変わり', '保留']):
            return random.choice(HIGH_TENSION_HAI_VARIATIONS), "待機要求"
        if any(kw in utterance_lower
               for kw in ['ですか', 'ますか', 'でしょうか', '?', '？']):
            return random.choice(HIGH_TENSION_HAI_VARIATIONS * 5 +
                                 ["確認します"] * 2), "質問形式"
        if any(kw in utterance_lower
               for kw in ['わかり', '了解', 'OK', 'オッケー', '承知', 'かしこまり']):
            return random.choice(HIGH_TENSION_HAI_VARIATIONS), "同意"
        if any(kw in utterance_lower
               for kw in ['違う', 'ちがう', 'いや', 'いいえ', 'そうじゃなくて']):
            return "失礼いたしました", "否定"
        return random.choice(HIGH_TENSION_HAI_VARIATIONS), "デフォルト"

    def process_conversation(self, user_text: str, history: list):
        history_text_list = []
        for h in history[-10:]:
            role_label = "店員" if h['role'] == 'user' else "AI"
            history_text_list.append(f"{role_label}: {h['text']}")
        history_text = "\n".join(history_text_list)

        prompt = f"""
**【最重要役割】あなたは、リストランテ鈴木の予約を電話で行っている「山田太郎のAIアシスタント」です。あなたは店員ではありません。**

以下の予約情報を元に、店員と会話を進めてください。

【予約情報】
- 予約者名: {self.RESERVATION_INFO['reserver_name']}
- 連絡先: {self.RESERVATION_INFO['contact_phone']}
- 希望日: {self.RESERVATION_INFO['date']}
- 希望時間: {self.RESERVATION_INFO['time']}
- 人数: {self.RESERVATION_INFO['guests']}名

【絶対に禁止】
- 店員側のセリフ（「何時がよろしいですか？」等）を絶対に言わない。
- 「承知いたしました」「かしこまりました」「了解いたしました」は絶対に使用しない。
- 応答にカッコ書き（例：「（電話を切る）」）を絶対に含めない。

---
### 【イレギュラー対応ルール：優先度 最上位】
---
1. **予約完了後の終話**
   - 店員が「お待ちしております」「承りました」等、予約成立を示した場合：「承知しました。本日はありがとうございました。失礼いたします。」と返し、終了する。

2. **予約不可・満席への対応**
   - 満席時：「承知しました。{self.RESERVATION_INFO['date']}の{self.RESERVATION_INFO['time']}から{self.RESERVATION_INFO['guests']}名では満席ということですね。残念ですが、今回はこれで失礼させていただきます。ありがとうございました。」

3. **本人確認/AI拒否への対応**
   - 拒否されたら一度だけ空き状況の交渉を行い、それでもダメなら「本人に伝えます。失礼しました」と引く。

4. **疎通困難・不明な質問（パターン対応）**
   - **【重要】** 単なる挨拶（「お電話ありがとう」「もしもし」）に対してこのルールを適用してはいけません。
   - **同じ質問を3回繰り返しても話が通じない場合や、意味不明なノイズが続く場合のみ**、「申し訳ありません、お電話が少し遠いようですので、一度失礼して改めてご連絡差し上げます。」と伝えて終話してください。
   - 予約に関係ない雑談をされた場合は、一度だけ「予約をお願いしたいのですが」と本題に戻してください。

---
### 【情報の出し方：重要】
---
- **積極的なリード**: 店員が挨拶（「もしもし」「はい、〇〇店です」）をした直後や、単なる相槌（「はい」）を打った場合は、待たずに「12月25日の予約をお願いしたいのですが」や「19時から4名です」と、こちらから情報を提示してリードしてください。
- **聞かれたことに答える**: 具体的な質問には正確に答える。情報提供時に「わかりました」等の同意語を混ぜない。
- **電話番号**: 復唱が一部不一致なら下4桁を言い直す。訂正は1回まで。
- **沈黙の活用**: 全ての情報を出し切り、店員が確認作業に入っている場合のみ、無言（空文字列 ""）を選択する。

【これまでの会話】{history_text}
【店員の発言】{user_text}
【あなたの応答】:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
        except ResourceExhausted:
            return "承知いたしました。本日はありがとうございました。失礼いたします。", history
        except Exception:
            return "申し訳ありません、お電話が少し遠いようですので、また改めてご連絡させていただきます。失礼いたします。", history

        if response_text.startswith("AI:"):
            response_text = re.sub(r'^AI:\s*', '', response_text)

        # カッコ書きを除去
        response_text = re.sub(r'（[^）]+）', '', response_text).strip()

        # 空文字列（沈黙）のハンドリング
        if response_text == "":
            return "", history

        new_history = history + [{
            "role": "user",
            "text": user_text
        }, {
            "role": "ai",
            "text": response_text
        }]
        return response_text, new_history
