import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QueryProcessor:
    def __init__(self):
        """
        Module xử lý câu truy vấn nâng cao cho Thành viên 1:
        1. Dịch câu hỏi tiếng Việt sang tiếng Anh Visual miêu tả chi tiết (dùng Helsinki Opus-MT vi-en offline).
        2. Nhận diện ý định câu hỏi (Intent Recognition): Phân loại câu hỏi dạng OCR/Text hay Visual Scene
           để điều chỉnh trọng số RRF Fusion linh hoạt.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "Helsinki-NLP/opus-mt-vi-en"
        
        print("QueryProcessor: Đang nạp mô hình dịch thuật Tiếng Việt -> Tiếng Anh...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.translator_available = True
            print("QueryProcessor: ✅ Đã khởi tạo thành công bộ dịch Tiếng Việt -> Tiếng Anh offline.")
        except Exception as e:
            print(f"QueryProcessor: Cảnh báo không nạp được Opus-MT ({e}), sử dụng chế độ fallback giữ nguyên query.")
            self.translator_available = False

    def translate_vi_to_en(self, text_vi):
        """Dịch câu truy vấn Tiếng Việt sang Tiếng Anh để tối ưu mô hình CLIP/SigLIP."""
        if not self.translator_available or not text_vi.strip():
            return text_vi
            
        try:
            inputs = self.tokenizer(text_vi, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                translated_tokens = self.model.generate(**inputs, max_length=100)
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            return text_vi

    def detect_query_intent(self, text_vi):
        """
        Nhận diện ý định câu hỏi để gán trọng số RRF Fusion:
        - Nếu chứa từ khóa như 'biển báo', 'chữ', 'bảng hiệu', 'số xe', 'viết từ' -> Ưu tiên BM25 OCR.
        - Ngược lại -> Ưu tiên Vector Search (CLIP/SigLIP).
        """
        text_lower = text_vi.lower()
        ocr_keywords = [
            "chữ", "biển báo", "bảng hiệu", "tên", "số xe", "biển số", 
            "dòng chữ", "viết", "ký tự", "bảng tên", "áo ghi số", "logo"
        ]
        
        for kw in ocr_keywords:
            if kw in text_lower:
                return {
                    "intent": "OCR_TEXT",
                    "dense_weight": 0.3,
                    "sparse_weight": 0.7
                }
                
        return {
            "intent": "VISUAL_SCENE",
            "dense_weight": 0.7,
            "sparse_weight": 0.3
        }

    def process(self, query_vi):
        """
        Xử lý toàn diện câu truy vấn đầu vào.
        Trả về dictionary chứa:
        - query_vi: Câu gốc tiếng Việt
        - query_en: Câu đã dịch sang tiếng Anh cho CLIP
        - intent_info: Thông tin ý định và trọng số gộp điểm
        """
        query_en = self.translate_vi_to_en(query_vi)
        intent_info = self.detect_query_intent(query_vi)
        
        return {
            "query_vi": query_vi,
            "query_en": query_en,
            "intent_info": intent_info
        }
