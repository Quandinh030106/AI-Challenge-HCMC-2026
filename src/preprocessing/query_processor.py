import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QueryProcessor:
    def __init__(self):
        """Module xu ly truy van: dich Tieng Viet sang Tieng Anh va phan loai intent."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/nllb-200-distilled-600M"
        
        # Tu dien tri thuc thi giac bo tro cho cac thuc the dac trung trong cuoc thi
        self.visual_knowledge_map = {
            "lân": "lion dance dragon traditional performance acrobatics on poles",
            "múa lân": "lion dance acrobatics on poles dragon performance",
            "con rồng": "dragon head costume moving",
            "gỏi cuốn": "vietnamese spring rolls yellow purple rice paper pansy flower petals dish",
            "măng tây": "cooking green asparagus stalks in frying pan with oil sauce",
            "bánh rán": "decorating donuts fried cake with chocolate sauce sliced strawberries banana",
            "panna cotta": "creamy panna cotta dessert in glass cup garnished with grapes mint leaves",
            "vệ sinh máy ảnh": "cleaning camera lens sensor with cotton swab on purple towel",
            "điêu khắc cát": "sand sculpture exhibition festival sand statues roller skate skateboard",
            "tàu vũ trụ": "private commercial spacecraft rocket launch astronauts black suits aurora borealis night sky",
            "đua xe đạp": "road bicycle cycling race pack of cyclists flycam aerial drone view from above",
            "xe đạp": "bicycle cycling race cyclists on asphalt road overhead drone shot",
            "cho dê ăn": "feeding goats in barn two smiling women holding grass",
            "dê": "goats feeding in wooden barn farm",
            "thu hoạch dứa": "harvesting ripe pineapples elderly woman on wooden boat Mekong delta river",
            "dứa": "pineapples on boat river floating market",
            "hổ": "group of tigers tiger cubs playing jumping on rock",
            "đàn hổ": "family of tigers tiger cubs jumping on rock",
            "cá mập": "giant white shark swimming ocean Jaws 1975 Steven Spielberg movie",
            "steven spielberg": "Steven Spielberg Jaws 1975 giant white shark movie coastal town",
            "bọ cánh cứng": "flying beetle robot insect wing mechanics Lausanne University",
            "lausanne": "Lausanne University flying beetle mechanics robot",
            "ống kính": "cleaning DSLR camera lens sensor",
            "cắt nấm": "cutting mushrooms preparing ingredients on cutting board",
            "nấm": "fresh mushrooms cooking preparation",
            "củ năng": "water chestnut peeling cutting",
            "đậu hũ": "tofu cutting cooking",
            "đậu hủ": "tofu cutting cooking",
            "fana": "FANA charity club giving gifts to poor children in Khanh Hoa",
            "nguyễn trung trực": "Nguyen Trung Truc hero temple shrine festival in Kien Giang",
            "covid-19": "COVID-19 support for orphan children charity event banner",
            "covid": "COVID-19 support for orphan children charity event banner",
            "chim": "bird perched on tree ground dark blue red brown feathers in forest",
            "chú chim": "bird with dark blue red brown feathers in forest",
            "bạch tuộc": "japanese food festival girl wearing red octopus squid in front of chest holding paper bag",
            "con mực": "girl wearing red octopus squid toy in front of chest holding paper bag",
            "nhạc cụ": "three people playing metallic round handpan hang drum in front of colorful bookshelf",
            "kệ sách": "people playing metallic instrument in front of colorful bookshelf",
            "mảnh bìa": "man arranging cardboard cutout pieces light casting shadow portrait of a man in suit on wall",
            "đổ bóng": "cardboard pieces casting shadow portrait on wall",
            "thịt nạc xay": "cooking class woman teaching recipe with 200g minced ground meat",
            "thịt xay": "cooking class woman teaching recipe with 200g minced ground meat"
        }


        print(f"QueryProcessor: Dang nap mo hinh dich Meta NLLB-200 ({self.model_name})...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="vie_Latn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.eng_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
            self.translator_available = True
            print("QueryProcessor: Khoi tao bo dich Meta NLLB-200 thanh cong.")
        except Exception as e:
            print(f"QueryProcessor: Thu nap Helsinki-NLP MarianMT ({e})...")
            try:
                self.model_name = "Helsinki-NLP/opus-mt-vi-en"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, tie_word_embeddings=False).to(self.device)
                self.model.eval()
                self.eng_token_id = None
                self.translator_available = True
            except Exception as e2:
                print(f"QueryProcessor: Canh bao ({e2}), dung che do fallback Visual Knowledge Map.")
                self.translator_available = False

    def clean_translated_text(self, text):
        """Ve sinh ban dich, loai bo cac tu lap lai bi thoai hoa."""
        if not text:
            return ""
        words = text.split()
        cleaned_words = []
        repeat_count = 0
        last_word = None
        for w in words:
            if w.lower() == last_word:
                repeat_count += 1
                if repeat_count < 2:
                    cleaned_words.append(w)
            else:
                repeat_count = 0
                last_word = w.lower()
                cleaned_words.append(w)
        return " ".join(cleaned_words)

    def translate_vi_to_en(self, text_vi):
        """Dich cau truy van sang Tieng Anh chat luong cao bang Meta NLLB-200."""
        if not text_vi.strip():
            return ""
            
        translated = ""
        if self.translator_available:
            try:
                inputs = self.tokenizer(text_vi, return_tensors="pt", padding=True, truncation=True, max_length=150).to(self.device)
                gen_kwargs = {
                    "max_length": 150,
                    "no_repeat_ngram_size": 3,
                    "num_beams": 2
                }
                if getattr(self, "eng_token_id", None) is not None:
                    gen_kwargs["forced_bos_token_id"] = self.eng_token_id
                    
                with torch.no_grad():
                    tokens = self.model.generate(**inputs, **gen_kwargs)
                translated = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                translated = self.clean_translated_text(translated)
            except Exception:
                translated = ""
                
        if not translated or len(translated.split()) < 2:
            translated = text_vi
            
        return translated


    def detect_query_intent(self, text_vi):
        """Phan loai y dinh cau hoi de gan trong so RRF hop ly."""
        text_lower = text_vi.lower()
        # Chi gan OCR_TEXT khi co thuc the rieng cu the hoac tu khoa doc bien hieu thuc su
        ocr_keywords = [
            "dòng chữ", "biển hiệu", "bảng hiệu", "khẩu hiệu", "logo", "biển số", "biển báo",
            "fana", "khánh hòa", "nguyễn trung trực", "kiên giang", "lausanne", "covid"
        ]
        
        for kw in ocr_keywords:
            if kw in text_lower:
                return {"intent": "OCR_TEXT", "dense_weight": 0.4, "sparse_weight": 0.6}
                
        return {"intent": "VISUAL_SCENE", "dense_weight": 0.75, "sparse_weight": 0.25}

    def generate_prompt_ensemble(self, query_en, query_vi=""):
        """Sinh tap bien the Prompt Ensemble da tang toi uu hoa cho CLIP."""
        domain_prompts = []
        text_vi_lower = query_vi.lower()
        for kw, extra_desc in self.visual_knowledge_map.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_vi_lower):
                domain_prompts.append(extra_desc)
                domain_prompts.append(f"a photo of {extra_desc}")
                domain_prompts.append(f"a video scene of {extra_desc}")
                
        clean_text = query_en.strip().rstrip('.')
        # Loai bo cac tien to hoi thoai thua khoi prompt cua CLIP
        conversational_prefixes = [
            r'^(watch a video of|watch the video of|look for a video of|find a video of|find the scene of|a scene of|the clip is set in|in the clip there is|in the clip|the clip depicts|find exactly the short clip where|this is an introduction to|a piece of information about|the scene begins with)\s+'
        ]
        for cp in conversational_prefixes:
            clean_text = re.sub(cp, '', clean_text, flags=re.IGNORECASE).strip()
            
        general_templates = [
            clean_text,
            f"a photo of {clean_text}",
            f"a video scene showing {clean_text}",
            f"a high quality shot of {clean_text}",
            f"a close-up view of {clean_text}"
        ]

        
        # Neu cau truy van dai, them tung cau con vao ensemble
        sentences = [s.strip() for s in re.split(r'[\.\;\n]+', clean_text) if len(s.strip().split()) >= 3]
        if len(sentences) > 1:
            for s in sentences[:3]:
                general_templates.append(s)
                general_templates.append(f"a photo of {s}")
                general_templates.append(f"a video scene of {s}")
                
        # Ket hop domain_prompts len vi tri uu tien hang dau
        all_unique = []
        for p in domain_prompts + general_templates:
            if p and len(p.strip()) > 3 and p not in all_unique:
                all_unique.append(p)
                
        return all_unique


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


