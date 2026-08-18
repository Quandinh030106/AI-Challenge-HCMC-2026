from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QueryProcessor:
    def __init__(self):
        """Module xu ly truy van: dich Tieng Viet sang Tieng Anh va phan loai intent."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "Helsinki-NLP/opus-mt-vi-en"
        
        print("QueryProcessor: Dang nap mo hinh dich Tieng Viet -> Tieng Anh...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.translator_available = True
            print("QueryProcessor: Khoi tao bo dich offline thanh cong.")
        except Exception as e:
            print(f"QueryProcessor: Canh bao ({e}), dung che do fallback giu nguyen query.")
            self.translator_available = False

    def translate_vi_to_en(self, text_vi):
        """Dich cau truy van sang Tieng Anh."""
        if not self.translator_available or not text_vi.strip():
            return text_vi
            
        try:
            inputs = self.tokenizer(text_vi, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                tokens = self.model.generate(**inputs, max_length=100)
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        except Exception:
            return text_vi

    def detect_query_intent(self, text_vi):
        """Phan loai y dinh cau hoi de gan trong so RRF."""
        text_lower = text_vi.lower()
        ocr_keywords = ["chữ", "biển báo", "bảng hiệu", "tên", "số xe", "biển số", "dòng chữ", "viết", "ký tự", "logo"]
        
        for kw in ocr_keywords:
            if kw in text_lower:
                return {"intent": "OCR_TEXT", "dense_weight": 0.3, "sparse_weight": 0.7}
                
        return {"intent": "VISUAL_SCENE", "dense_weight": 0.7, "sparse_weight": 0.3}

    def generate_prompt_ensemble(self, query_en):
        """Sinh tap bien the Prompt Ensemble."""
        clean_text = query_en.strip().rstrip('.')
        templates = [
            clean_text,
            f"a photo of {clean_text}",
            f"a video scene showing {clean_text}",
            f"a close-up shot of {clean_text}",
            f"a wide angle view of {clean_text}"
        ]
        return list(set(templates))

    def process(self, query_vi):
        """Xu ly toan dien cau truy van."""
        query_en = self.translate_vi_to_en(query_vi)
        prompt_ensemble = self.generate_prompt_ensemble(query_en)
        intent_info = self.detect_query_intent(query_vi)
        
        return {
            "query_vi": query_vi,
            "query_en": query_en,
            "prompt_ensemble": prompt_ensemble,
            "intent_info": intent_info
        }
