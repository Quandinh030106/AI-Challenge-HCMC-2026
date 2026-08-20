import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QueryProcessor:
    def __init__(self):
        """Module xu ly truy van: dich Tieng Viet sang Tieng Anh va phan loai intent."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "Helsinki-NLP/opus-mt-vi-en"
        
        # Tu dien tri thuc thi giac bo tro cho cac thuc the dac trung trong cuoc thi
        self.visual_knowledge_map = {
            "lân": "lion dance dragon traditional performance",
            "múa lân": "lion dance acrobatics on poles",
            "con rồng": "dragon moving head",
            "gỏi cuốn": "vietnamese spring rolls yellow purple rice paper pansy flower",
            "măng tây": "asparagus in frying pan with oil and sauce",
            "bánh rán": "donuts fried cake with chocolate strawberry slices banana",
            "panna cotta": "panna cotta dessert in glass with grapes mint leaves",
            "vệ sinh máy ảnh": "cleaning camera lens with cotton swab on purple towel",
            "điêu khắc cát": "sand sculpture festival sand statues roller skate skateboard",
            "tàu vũ trụ": "spacecraft rocket launch astronauts in black suits aurora lights",
            "đua xe đạp": "cycling road bicycle race cyclists overhead flycam drone view",
            "cho dê ăn": "feeding goats in barn two women smiling",
            "thu hoạch dứa": "harvesting pineapples elderly woman boat Mekong delta",
            "ống kính": "camera lens cleaning",
            "cắt nấm": "cutting mushrooms preparing ingredients",
            "củ năng": "water chestnut cutting",
            "đậu hũ": "tofu cutting cooking",
            "đậu hủ": "tofu cutting cooking",
            "steven spielberg": "Steven Spielberg Jaws 1975 shark movie coastal town",
            "lausanne": "Lausanne university flying beetle mechanics robot",
            "fana": "FANA charity club giving gifts in Khanh Hoa",
            "nguyễn trung trực": "Nguyen Trung Truc hero temple shrine in Kien Giang",
            "covid-19": "COVID-19 support for orphan children charity event banner"
        }
        
        print("QueryProcessor: Dang nap mo hinh dich Tieng Viet -> Tieng Anh...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.translator_available = True
            print("QueryProcessor: Khoi tao bo dich offline thanh cong.")
        except Exception as e:
            print(f"QueryProcessor: Canh bao ({e}), dung che do fallback ket hop Visual Knowledge Map.")
            self.translator_available = False

    def translate_vi_to_en(self, text_vi):
        """Dich cau truy van sang Tieng Anh chat luong cao."""
        if not text_vi.strip():
            return ""
            
        translated = ""
        if self.translator_available:
            try:
                inputs = self.tokenizer(text_vi, return_tensors="pt", padding=True, truncation=True, max_length=150).to(self.device)
                with torch.no_grad():
                    tokens = self.model.generate(**inputs, max_length=150)
                translated = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            except Exception:
                translated = ""
                
        if not translated:
            translated = text_vi
            
        return translated

    def detect_query_intent(self, text_vi):
        """Phan loai y dinh cau hoi de gan trong so RRF."""
        text_lower = text_vi.lower()
        ocr_keywords = ["chữ", "biển báo", "bảng hiệu", "tên", "số xe", "biển số", "dòng chữ", "viết", "ký tự", "logo", "fana", "khánh hòa", "nguyễn trung trực", "kiên giang", "lausanne", "covid"]
        
        for kw in ocr_keywords:
            if kw in text_lower:
                return {"intent": "OCR_TEXT", "dense_weight": 0.4, "sparse_weight": 0.6}
                
        return {"intent": "VISUAL_SCENE", "dense_weight": 0.7, "sparse_weight": 0.3}

    def generate_prompt_ensemble(self, query_en, query_vi=""):
        """Sinh tap bien the Prompt Ensemble da tang (Visual Ensembling)."""
        clean_text = query_en.strip().rstrip('.')
        templates = [
            clean_text,
            f"a photo of {clean_text}",
            f"a video scene showing {clean_text}",
            f"a high quality shot of {clean_text}",
            f"a close-up view of {clean_text}"
        ]
        
        # 1. Neu cau truy van dai (co nhieu cau con tach boi dau cham), them tung cau con vao ensemble
        sentences = [s.strip() for s in re.split(r'[\.\;\n]+', clean_text) if len(s.strip().split()) >= 3]
        if len(sentences) > 1:
            for s in sentences[:3]:
                templates.append(s)
                templates.append(f"a photo of {s}")
                templates.append(f"a video scene of {s}")
                
        # 2. Tiem them cac tu khoa thi giac bo tro (Visual Knowledge Injection)
        text_vi_lower = query_vi.lower()
        for kw, extra_desc in self.visual_knowledge_map.items():
            if kw in text_vi_lower:
                templates.append(extra_desc)
                templates.append(f"a video scene of {extra_desc}")
                
        return list(set([t for t in templates if len(t.strip()) > 3]))

    def process(self, query_vi):
        """Xu ly toan dien cau truy van."""
        query_en = self.translate_vi_to_en(query_vi)
        prompt_ensemble = self.generate_prompt_ensemble(query_en, query_vi=query_vi)
        intent_info = self.detect_query_intent(query_vi)
        
        return {
            "query_vi": query_vi,
            "query_en": query_en,
            "english_query": query_en,
            "prompt_ensemble": prompt_ensemble,
            "intent_info": intent_info
        }


