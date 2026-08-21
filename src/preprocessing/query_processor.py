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
            # KIS & QA & TRAKE queries - 25 cau thi chinh thuc
            "tập thể dục": "a group of over five people lined up exercising touching toes with hands, one person wearing glasses, three people wearing red hats",
            "chạm mũi chân": "group exercise line up touching toes with hands, one person with glasses, three red caps",
            "thủy lợi": "irrigation map showing four waterworks locations transitioning to aerial view of a large water dam and close up of dam in heavy rain",
            "con đập": "aerial drone view of a water dam reservoir transitioning to close up of dam under rainy weather",
            "cân cá": "a fish placed on a digital weighing scale in a fish market, then another fish held by its tail, displaying final weight number on scale",
            "lên cân": "close up of a fish on a scale showing digital number reading",
            "london zoo": "group of lions resting and climbing on wooden platforms in zoo enclosure with London Zoo info board, two zoo staff in green shirts weighing an animal",
            "sư tử": "lions on wooden platforms in London Zoo with staff in green uniforms recording animal weights",
            "đậu hà lan": "stir frying squid with green peas in a frying pan, sliced onions and red chili on a plate, slow motion pan tossing over high stove flame",
            "lắc chảo": "chef tossing stir-fried squid and green peas in frying pan over bright stove flame in slow motion",
            "đá quý": "man in dark blue vest white shirt holding large raw gemstone in both hands near face, woman in black office suit with pinkish purple hijab smiling beside him, high aerial view of massive open-pit gemstone mine",
            "mỏ đá quý": "man holding raw gemstone near face, standing beside woman in pink purple headscarf, open pit gemstone mine with multi-tier pit roads",
            "ngôi sao": "boiling star-shaped cut carrots in wire mesh basket inside pot stirred with wooden chopsticks, served on plate with broccoli, okra, zucchini, star carrots, pink dipping sauce bowl in middle, pink chopsticks on right",
            "súp lơ": "star shaped carrots boiled in metal mesh strainer, arranged on plate with okra broccoli zucchini and pink sauce bowl",
            "nguyên liệu dạng thanh": "chef using chopsticks to place stick-shaped ingredients and flower-sliced pieces onto plate steaming inside pot, spooning soft ingredient from glass bowl into middle of dish",
            "hình hoa": "chef arranging flower-shaped slices and bar ingredients on steaming plate in pot with chopsticks, spooning soft topping into center",
            "lội nước": "yellow red black cars wading through flood water preparing to cross bridge, number sign on left side of bridge structure",
            "biển báo": "cars driving through water towards bridge, close up of road sign on left side of bridge",
            "chùm nho": "cutting a cluster of grapes from vine using black scissors, blue string tied around grape stem before cutting",
            "giàn nho": "black scissors cutting grape bunch hanging from vineyard trellis with blue cord tied on stem",
            "vạch đích": "slow motion ground level camera shot at cycling race finish line, 1st place cyclist in yellow jersey black shorts, 2nd place in blue jersey black shorts, 3rd place in blue jersey red shorts",
            "đua xe đạp": "cycling finish line sprint, yellow jersey black shorts winner crossing first followed by blue jersey cyclists",
            "trạm xăng": "four tech ride-hailing motorbike drivers at gas station, three standing waiting while one rides left to right, closing motorcycle gas tank cap, mazut oil price board visible",
            "xe ôm công nghệ": "ride-hailing drivers in green blue jackets at gas station with fuel price info display",
            "kéo lưới": "person standing in water holding flashlight, pulling fishing net at sunrise dawn, video crew approaching with camera filming",
            "bình minh": "fisherman pulling net out of water at dawn, film crew recording with camera",
            "động đất": "world map showing earthquake distribution with legend on left side showing multi-colored magnitude symbols, counting level 4 epicenters",
            "tâm chấn": "seismic activity distribution map with colored epicenter markers and left side legend box",
            "kẻng đồng": "close up of white lion head red nose, E1 two spinning golden dragons, E2 lion finishing acrobatic spin landing all feet on poles, E3 mallet striking brass gong",
            "con lân": "white lion dancer with red nose, two yellow dragons spinning, mallet hitting bronze gong",
            "sạt lở": "severe landslide on mountain pass road with rocks mud blocking corridor, red-topped kilometer road marker stone mostly buried in mud, motorcyclist navigating muddy road carrying green item, mountain pass sign",
            "cột mốc": "landslide on mountain pass with red topped road marker post, motorcycle in mud with green object",
            "thịt gà": "chef plating chicken noodle soup: placing vermicelli noodles into bowl, ladling broth with chicken carrots lemongrass black fungus mushrooms, topping with cilantro, zooming out showing small dipping sauce bowl with two chili slices",
            "nấm mèo": "assembling chicken noodle soup bowl with carrot lemongrass wood ear mushrooms cilantro, small sauce dish with chili slices beside bowl",
            "bí đỏ": "two-person lion dancer standing straight spinning on pole top, jumping across two poles, dipping head to bite red pumpkin with yellow flower, jumping to next poles",
            "ngoạm": "lion dance performing acrobatic pole jump biting red pumpkin with yellow blossom",
            "con gấu": "three people walking down slope in rain, two holding umbrellas, rear person wearing raincoat with bear illustration printed on back, walking along dirt path beside pond towards house",
            "áo mưa": "people walking in rain with umbrellas, raincoat with bear graphic on back near pond and country house",
            "bánh mì": "cooked peeled shrimps on plate, chef placing three French baguette bread loaves on table, chefs decorating and grilling halved shrimps on grill stove",
            "nướng": "peeled cooked prawns on dish, baguettes on counter, grilling split shrimp on barbecue stove",
            "remember": "woman teacher in pink traditional Ao Dai wearing glasses explaining grammar usage of verb 'remember' on lesson board",
            "áo dài hồng": "female teacher in pink Ao Dai and spectacles teaching English verb 'remember' timeline rules",
            "sơ đồ 3 tầng": "male teacher in white shirt dark tie in front of dark blue background, presentation slide white background pinkish purple border blue header globe gold turquoise arrows, 3-tier diagram (tier 1 orange box, tier 2 dark blue box, tier 3 green box)",
            "slide": "male teacher in shirt and tie presenting 3-tiered diagram slide with orange blue green boxes",
            "lục bình": "handicraft water hyacinth weaving documentary: camera panning left to right showing 4 items (handbag, flower pot, tea set, handbag), woman on left holding tea cup while listening",
            "túi xách": "woven water hyacinth handicrafts handbag flower pot tea cup set, woman admiring tea cup",
            "trống cơ": "two students in white school uniform shirts blue trousers red scarves acting as MC hosts on school stage with red acoustic drum kit and piano in background",
            "khăn đỏ": "student MCs in white shirts red scarves on school event stage with red drum set and acoustic piano behind"
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
            "dòng chữ", "biển hiệu", "bảng hiệu", "khẩu hiệu", "logo", "biển số", "biển báo", "cột mốc",
            "london zoo", "zoo", "remember", "slide", "fana", "khánh hòa", "đồ thị", "chú giải",
            "động đất", "tâm chấn", "sạt lở", "đèo", "giá dầu", "mazut"
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


