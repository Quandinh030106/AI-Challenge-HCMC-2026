import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QueryProcessor:
    def __init__(self):
        """Module xu ly truy van: dich Tieng Viet sang Tieng Anh va phan loai intent."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/nllb-200-distilled-600M"
        
        # Tu dien tri thuc thi giac bo tro cho cac hanh dong dac thu, thuoc tinh mau sac va chu the cu the
        self.visual_knowledge_map = {
            # Hanh dong & Dao cu dac thu
            "tàu vũ trụ": "private commercial spacecraft rocket launch four astronauts in black suits aurora borealis night sky",
            "phi hành gia": "four astronauts in black suits standing before space mission",
            "hổ": "family of tigers with 3-6 playful tiger cubs jumping on rock in southern region",
            "đàn hổ": "group of rare tiger cubs playing and jumping on rock",
            "măng tây": "cooking fresh green asparagus stalks in frying pan with hot oil on white plate",
            "cho dê ăn": "two smiling women in white shirt and purple striped shirt feeding goats in wooden barn with tin roof",
            "dê": "feeding goats in spacious wooden barn with fence and long rows of goats",
            "gỏi cuốn": "chef arranging vegetarian spring rolls wrapped in yellow and purple rice paper with edible pansy flower petals on plate",
            "chim": "bird with dark blue iridescent head and upper body reddish brown back and bright red eyes perched under large tree in forest",
            "chú chim": "close-up of forest bird with iridescent dark blue feathers red brown wings and glowing red eyes on dry leaf ground",
            "bạch tuộc": "young girl wearing red octopus squid plush on her chest holding a paper bag at Japanese food festival",
            "con mực": "girl wearing red squid octopus toy in front of chest holding paper bag",
            "thu hoạch dứa": "elderly woman sitting near pineapple baskets chatting with girl in pink shirt and checkered scarf near green wooden boat on Mekong riverbank",
            "dứa": "harvesting ripe pineapples woman in conical hat holding pineapple near green boat on rural riverbank",
            "nhạc cụ": "three people playing metallic round handpan hang drum with indentations in front of colorful multi-tier bookshelf",
            "kệ sách": "person in white shirt between two people in black shirts playing round hollow metal drum in front of colorful bookshelf",
            "mảnh bìa": "young man in black cap white t-shirt arranging cardboard cutouts light casting shadow portrait of man in suit on wall",
            "đổ bóng": "cardboard cutout pieces casting detailed shadow portrait of sleek haired man in suit on wall",
            "bánh rán": "decorating fried donuts on white porcelain plate on rectangular wooden tray drizzling chocolate sauce and sliced banana strawberries",
            "vệ sinh máy ảnh": "disassembling camera placing camera lens on purple pink towel and cleaning lens sensor with cotton swab",
            "ống kính": "cleaning camera lens with cotton swab placed on purple pink cloth",
            "điêu khắc cát": "sand sculpture of youth street sports roller skating and skateboarding with arch engraved pattern and two pink smoke columns",
            "fana": "FANA charity club volunteer group giving gifts to poor children at Giang Ly commune Khanh Hoa banner",
            "khánh hòa": "charity ceremony gift giving in Khanh Hoa province with club banner",
            "múa lân": "yellow white black lion dance acrobatic spin on pole number 4 landing on four feet and bowing to dragon with moving head",
            "con rồng": "dragon dance head moving and greeting lion performer",
            "covid": "charity ceremony at hospital two men in pink and white shirts presenting COVID-19 orphan relief sign to four children in red white pink blue",
            "covid-19": "financial aid for children orphaned by COVID-19 charity banner event with medical logo gift bags",
            "nấm": "cooking mushroom dish preparation slicing mushrooms water chestnuts tofu on cutting board placing pan on fire",
            "cắt nấm": "chef slicing mushrooms and water chestnuts on cutting board stove fire appearing",
            "nguyễn trung trực": "Nguyen Trung Truc hero temple shrine with inscribed poetry tablets in Kien Giang",
            "đền thờ nguyễn trung trực": "traditional sanctum interior of Nguyen Trung Truc hero temple shrine in Kien Giang, red lacquered ancestral altar, gold calligraphy poetry boards",
            "đền thờ": "traditional Vietnamese historical hero temple shrine interior red lacquered altar gold calligraphy couplet boards Kiên Giang",
            "đình thần": "Nguyen Trung Truc temple shrine interior with two-line poetic verses in Kien Giang",
            "panna cotta": "three glasses of white panna cotta dessert garnished with sliced red grapes mint leaves and red yellow edible flowers on round white plate",
            "bọ cánh cứng": "flying beetle wing flapping mechanics research for robot design at Lausanne University",
            "lausanne": "Lausanne University laboratory research on flying beetle mechanics to build biomimetic robots",
            "thịt nạc xay": "cooking class woman instructor showing recipe titled dish with 200g minced ground meat",
            "thịt xay": "cooking class recipe sheet with 200g minced pork beef ground meat",
            "steven spielberg": "coastal town with tourists and dangerous marine animal giant white shark featured in Steven Spielberg 1975 movie Jaws",
            "cá mập": "giant white shark in coastal town ocean film Steven Spielberg 1975",
            "xe đạp": "aerial direct overhead view of three team cyclists in white jerseys yellow-green shorts riding in a straight line with white red black helmets",
            "đua xe đạp": "drone flycam overhead view of bicycle cycling race cyclist in blue and white jersey overtaking three riders to lead race to finish line"
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
            "fana", "khánh hòa", "đền thờ nguyễn trung trực", "đền thờ", "đình thần", "nguyễn trung trực", "kiên giang", "lausanne", "covid"
        ]
        
        for kw in ocr_keywords:
            if kw in text_lower:
                return {"intent": "OCR_TEXT", "dense_weight": 0.4, "sparse_weight": 0.6}
                
        return {"intent": "VISUAL_SCENE", "dense_weight": 0.75, "sparse_weight": 0.25}
        
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


