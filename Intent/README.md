# Hướng dẫn tạo và huấn luyện model Intent Detection cho trợ lý AI du lịch

Thư mục này chứa đầy đủ pipeline từ việc sinh dữ liệu (sử dụng Gemini API) cho tới việc finetune và test model:

## 1. Các file trong thư mục:
- `generate_dataset.py`: Mã nguồn gọi Gemini-3.0-flash-preview theo API Key trong file `.env` để tạo ra bộ data với tổng cộng 3100 câu phân mảnh theo 8 intents khác nhau. Output sẽ lưu ra file `intent_dataset.jsonl`.
- `train_intent.py`: Mã nguồn load dataset và Finetune một LLM/Encoder nhẹ `vinai/phobert-base-v2` (tầm ~135M parameters, rất xuất sắc với Tiếng Việt).
- `inference.py`: Mã nguồn test kết quả thực tế trên cmd với model đã pretrained.

## 2. Cách chạy:

### Bước 1: Sinh Dataset
Đảm bảo bạn đã cài `requests` và `python-dotenv`:
```bash
python generate_dataset.py
```
> **Lưu ý**: Lệnh này sẽ mất vài phút do phải gọi API nhiều lần để tạo ra hơn 3000 mẫu câu chất lượng.

### Bước 2: Huấn luyện mô hình
Chỉ cần chạy lệnh sau để tiến hành huấn luyện (Finetune):
```bash
python train_intent.py
```
Sau 5 epochs, mô hình tốt nhất sẽ được lưu vào `.Intent/phobert-intent-model`.

### Bước 3: Kiểm thử
Chạy file inference để chat trực tiếp và xem khả năng nhận diện intent của mô hình:
```bash
python inference.py
```

## 3. Các Intents được hỗ trợ
1. `travel_query` (500 câu): Yêu cầu gợi ý điểm đến/du lịch chung.
2. `itinerary_request` (400 câu): Yêu cầu lịch trình chi tiết (có số ngày).
3. `budget_query` (400 câu): Hỏi về chi phí, giá tiền.
4. `accommodation` (400 câu): Hỏi về khách sạn, nơi lưu trú.
5. `food_recommendation` (400 câu): Hỏi quán ăn, món ăn, đặc sản.
6. `preference_update` (300 câu): Cập nhật sở thích (thích biển, ghét ồn ào...).
7. `negative_feedback` (300 câu): Phản hồi không tốt hoặc báo lỗi (sai rồi, chán).
8. `chit_chat` (300 câu): Giao tiếp hàng ngày (hi, hello).
