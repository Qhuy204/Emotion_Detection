import os
import json
import time
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv("/home/qhuy/Emotion_detection/.env")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

MODEL_NAME = "gemini-3-flash-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

ERROR_CASES = """
BẮT BUỘC MỖI BATCH DATA SINH RA PHẢI BAO GỒM ĐẠI DIỆN TẤT CẢ TỈ LỆ LỖI VÀ ĐẶC ĐIỂM DƯỚI ĐÂY (PHÂN BỔ ĐỀU CHO TỰ NHIÊN NHƯ THẬT, RẤT QUAN TRỌNG):
1. Lỗi chính tả do gõ nhanh/không dấu: "ko", "k", "khog", "dc", "mk", "mik", "vs"...
2. Nhầm lẫn phụ âm đầu: s/x (ví dụ: xung sướng), ch/tr (chung thành), d/gi/r (gia đình rùi), l/n (nàm nũng).
3. Nhầm hỏi - ngã: "nghỉa vụ", "giử gìn", "đả", "sẻ"...
4. Thiếu hoặc sai dấu thanh: "toi muon gap ban", "cảm on", "đi hoc"...
5. Viết hoa/viết thường tùy tiện: "chào Anh", "em YÊU anh", "hà nội", "việt nam"...
6. Lặp chữ, kéo dài chữ: "trờiiiiiiii", "điiiiii", "hiiiiii"...
7. Thiếu chủ ngữ/câu cụt: "Đang làm.", "Mai đi."
8. Dùng từ sai nghĩa do phát âm vùng miền: ví dụ "bàng quang" thay vì "bàng hoàng", "trân thành" thay vì "chân thành".
9. Lỗi autocorrect: "đi ăn cơm" thành "đi ăn cơm chó", "ok" thành từ lạ...
10. Thiếu dấu câu: Câu dài viết dính liền không dấu phẩy, dấu chấm cản trở đọc như: "em ăn cơm chưa anh đi làm về rồi"
11. Từ ngữ địa phương, tiếng lóng, sinh viên, người lớn tuổi... (đa dạng vùng miền).
12. Lỗi gõ telex/vni (unicode): "với" => voiws, "không" => khoong, "đường" => dduwowngf.

Phải sinh NGẪU NHIÊN mix giữa các lỗi này, và có những câu CHUẨN MỰC xen kẽ.
"""

INTENTS = [
    {
        "name": "travel_query",
        "description": "Hỏi gợi ý địa điểm du lịch hoặc hỏi nên đi đâu, nhưng chưa yêu cầu lịch trình chi tiết hay hỏi giá cụ thể. Bao gồm: hỏi nên đi đâu, hỏi địa điểm theo khu vực, hỏi theo mùa, hỏi theo vibe (chill, mát mẻ, yên tĩnh...).",
    },
    {
        "name": "itinerary_request",
        "description": "Yêu cầu lịch trình cụ thể: đi mấy ngày, lịch trình chi tiết từng ngày, kế hoạch tham quan. KHÔNG bao gồm câu chỉ hỏi địa điểm chung.",
    },
    {
        "name": "budget_query",
        "description": "Hỏi về chi phí, giá tiền, ngân sách, giá tour, giá khách sạn hoặc tổng chi phí chuyến đi. Bao gồm hỏi giá cụ thể, ngân sách đủ không, dưới X tiền đi đâu được.",
    },
    {
        "name": "accommodation",
        "description": "Hỏi về khách sạn, resort, homestay, chỗ ở qua đêm, đặt phòng.",
    },
    {
        "name": "food_recommendation",
        "description": "Hỏi ăn gì, đặc sản, quán ăn, ẩm thực địa phương. KHÔNG bao gồm chỗ ở, giá tour.",
    },
    {
        "name": "preference_update",
        "description": "Cập nhật sở thích cá nhân: ví dụ thích biển hơn núi, không thích chỗ đông người, muốn nơi yên tĩnh, ghét ồn ào.",
    },
    {
        "name": "negative_feedback",
        "description": "Phản hồi tiêu cực về gợi ý hoặc câu trả lời của chatbot: nói sai, không đúng ý, chê, thất vọng.",
    },
    {
        "name": "chit_chat",
        "description": "Giao tiếp xã giao: chào hỏi, hỏi bot là ai, cảm ơn, tạm biệt.",
    }
]

TARGET_SAMPLES = 6000 # ~6000 samples per intent (within the 5000-8000 range)
BATCH_SIZE = 200 # Khoảng 200 câu một lần gọi để model không bị quá tải token (khoảng 2000-3000 tokens response)

def fetch_batch(intent_name, intent_desc, batch_idx, size):
    prompt = f"""Bạn là một chuyên gia sinh dữ liệu huấn luyện NLP tiếng Việt.

NHIỆM VỤ CỦA BẠN:
- Sinh chính xác {size} câu giao tiếp đóng vai người dùng thật đang chat với chatbot du lịch.
- Các câu này PHẢI thuộc Intent: "{intent_name}".
- Ý nghĩa Intent "{intent_name}": {intent_desc}

=== YÊU CẦU BẮT BUỘC VỀ ĐỘ ĐA DẠNG & LỖI (QUAN TRỌNG NHẤT) ===
Một bộ dữ liệu thực tế trên chatbot cần mix lộn xộn các câu chuẩn chỉ và các câu CỐ TÌNH MẮC LỖI thực tế.
{ERROR_CASES}
Lưu ý: Để đảm bảo không bị trùng ý, đây là đợt sinh thứ {batch_idx + 1}, hãy đổi hoàn toàn cách khai báo, ngữ cảnh, từ vựng so với trước. 

=== ĐẦU RA YÊU CẦU ===
- Trả về DUY NHẤT một mảng JSON hợp lệ chứa các câu chuỗi. KHÔNG chứa định dạng JSON Markdown ```, KHÔNG CÓ BẤT KỲ CÂU CHỮ NÀO KHÁC BÊN NGOÀI.
- Ví dụ MONG ĐỢI ĐẦU RA: ["câu 1", "câu 2", "câu 3"]
"""

    for attempt in range(5):
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 1.1 + (attempt * 0.1), # Tăng tính random qua mỗi lần retry hoặc sinh batch
                    "max_output_tokens": 8192
                }
            }
            
            response = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=240)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" not in result or not result["candidates"]:
                time.sleep(5)
                continue
                
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Xử lý dọn dẹp markdown
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            if isinstance(data, list):
                if len(data) < size * 0.5: # Quá ít, fail
                    continue
                return data
                
        except Exception as e:
            # Im lặng retry
            time.sleep(3)
            
    return []

def main():
    output_file = "/home/qhuy/Emotion_detection/Intent/intent_dataset_large.jsonl"
    
    # Đếm số lượng đã sinh để tiếp tục nếu bị gián đoạn
    counts = {intent["name"]: 0 for intent in INTENTS}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    counts[obj["label"]] += 1
                except:
                    pass
                    
    for intent in INTENTS:
        name = intent["name"]
        print(f"\n[{name}] >>> Đã có: {counts[name]}/{TARGET_SAMPLES} câu.")
        
        batch_idx = counts[name] // BATCH_SIZE
        
        while counts[name] < TARGET_SAMPLES:
            remaining = TARGET_SAMPLES - counts[name]
            request_size = min(BATCH_SIZE, remaining)
            
            print(f"[{name}] Gọi API batch {batch_idx + 1}... ({request_size} câu)")
            start = time.time()
            sentences = fetch_batch(name, intent["description"], batch_idx, request_size)
            
            if not sentences:
                print(f"[{name}] API Timeout hoặc lỗi nội bộ, retry sau 5s...")
                time.sleep(5)
                continue
            
            added = 0
            with open(output_file, "a", encoding="utf-8") as f:
                for sentence in sentences:
                    if isinstance(sentence, str) and sentence.strip():
                        item = {"text": sentence.strip(), "label": name}
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        added += 1
            
            counts[name] += added
            print(f"[{name}] +{added} câu thành công! ({time.time() - start:.1f}s) -> Tổng: {counts[name]}/{TARGET_SAMPLES}")
            batch_idx += 1
            time.sleep(1) # Chống spam

if __name__ == "__main__":
    main()
