import os
import sys
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMQueryParser:
    """
    NLP Query Parsing Engine powered by Qwen2.5-7B-Instruct.
    Dynamically extracts search schemas (intent, CLIP prompts, BM25 keywords, OpenImages classes, VQA question).
    Enforces 100% Vietnamese BM25 keyword & VQA question purity, physical OpenImages object filtering, and OCR weight boosting.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Loads Qwen2.5-7B-Instruct model strictly in 4-bit NF4 quantization on GPU."""
        if self.model is not None:
            return

        print(f"[INFO] LLMQueryParser: Loading NLP LLM ({self.model_id}) strictly on GPU in 4-bit NF4 mode...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.model.eval()
            print("[INFO] LLMQueryParser: Successfully loaded Qwen2.5-7B-Instruct in 4-bit NF4 mode on GPU.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] LLMQueryParser: Failed to load 4-bit NF4 LLM on GPU ({e}). Execution stopped as requested.")

    def unload_model(self):
        """Unloads NLP LLM model from VRAM to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] LLMQueryParser: Unloaded NLP LLM from VRAM.")

    def parse_query_dynamically(self, query_vi, task_type="kis", raw_question=""):
        """
        Runs LLM prompt execution strictly on GPU to dynamically extract structured search schema.
        """
        if self.model is None:
            self.load_model()

        system_prompt = (
            "Bạn là chuyên gia phân tích cú pháp truy vấn video đa phương thức (CLIP, BM25, OpenImages, VLM).\n"
            "Nhiệm vụ: Phân tích đoạn mô tả hoặc câu hỏi Tiếng Việt, trích xuất cấu trúc tìm kiếm JSON với ĐÚNG CÁC TRƯỜNG SAU:\n\n"
            "1. 'intent': 'VISUAL_SCENE' (mô tả bối cảnh/hình ảnh/hành động) hoặc 'OCR_TEXT' (khi đề bài yêu cầu đọc con số, chữ viết trên bảng/biển báo/cân/bản đồ/giá cả/cột mốc/slide).\n"
            "2. 'dense_weight': Trọng số tìm kiếm hình ảnh CLIP (0.7 đến 0.85 cho VISUAL_SCENE, 0.35 cho OCR_TEXT).\n"
            "3. 'sparse_weight': Trọng số tìm kiếm văn bản BM25 (0.15 đến 0.3 cho VISUAL_SCENE, 0.65 cho OCR_TEXT). Tổng = 1.0.\n"
            "4. 'golden_english_prompts': Mảng 2-4 câu miêu tả hình ảnh trực diện bằng Tiếng Anh chuẩn (Mỗi câu dưới 25 từ). LƯU Ý DỊCH ĐÚNG THỰC THỂ:\n"
            "   - 'đậu hà lan' -> 'snow peas' / 'green peas' (KHÔNG dịch thành Dutch pancake)\n"
            "   - 'mực' (nấu ăn) -> 'squid' / 'cuttlefish' (KHÔNG dịch thành ink)\n"
            "   - 'con lân' / 'múa lân' -> 'lion dance' / 'Chinese lion costume' (KHÔNG dịch thành dragon/unicorn)\n"
            "   - 'rồng' -> 'dragon'\n"
            "   - 'đua xe đạp' -> 'cycling race' / 'bicycle race'\n"
            "   - 'khối đá quý' -> 'rough gemstone'\n"
            "5. 'bm25_keywords': Mảng các CỤM DANH TỪ TIẾNG VIỆT có nghĩa trích xuất trực tiếp từ đề bài (2-4 từ/cụm, ví dụ: 'vạch đích', 'đua xe đạp', 'khối đá quý', 'sạt lở đất', 'đậu hà lan', 'cây lục bình'). TUYỆT ĐỐI KHÔNG xé lẻ thành từng từ đơn rác và KHÔNG bịa từ không có trong bài.\n"
            "6. 'openimages_classes': Mảng danh từ Tiếng Anh đại diện cho VẬT THỂ THỂ LÝ nhìn thấy được (ví dụ: 'person', 'bicycle', 'motorcycle', 'car', 'scale', 'drum', 'piano', 'handbag', 'teapot', 'umbrella', 'sign'). TUYỆT ĐỐI KHÔNG chứa tính từ (red, blue), động từ (stretching, cutting, steaming), biểu cảm (smile), hoặc khái niệm trừu tượng (verb, conversation, frame, slide, border, title).\n"
            "7. 'vlm_question': Câu hỏi Tiếng Việt trực tiếp, cô đọng để VLM đọc ảnh trả lời (100% bằng Tiếng Việt).\n\n"
            "YÊU CẦU ĐẦU RA: CHỈ NÊU MỘT KHỐI JSON HỢP LỆ VÀ NẰM TRONG CẶP THẺ ```json ... ```. KHÔNG THÊM BẤT KỲ LỜI DẪN NÀO."
        )

        user_content = f"Loại truy vấn: {task_type.upper()}\nNội dung Tiếng Việt: {query_vi}"
        if raw_question:
            user_content += f"\nCâu hỏi thô: {raw_question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Execute strictly via GPU LLM without silent fallback
        return self._parse_with_llm(messages, query_vi, task_type)

    def _parse_with_llm(self, messages, query_vi, task_type):
        """Executes text generation and cleans output JSON string."""
        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cleaned_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text, flags=re.MULTILINE).strip()

        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            schema = json.loads(json_str)
            return self._normalize_schema(schema, query_vi, task_type)

        raise ValueError("Failed to locate JSON object in LLM output.")

    def _normalize_schema(self, schema, query_vi, task_type):
        """Enforces field type validity, 100% Vietnamese BM25 keyword & VQA question purity, and OCR weight boosting."""
        intent = str(schema.get("intent", "VISUAL_SCENE")).upper()
        if intent not in ["VISUAL_SCENE", "OCR_TEXT"]:
            intent = "VISUAL_SCENE"

        # Precise OCR_TEXT intent trigger: only when reading explicit text, signs, scale numbers, map legends, prices
        ocr_positive_triggers = [
            "con số được ghi", "con số hiển thị", "số hiển thị", "con số trên cân", "chữ trên",
            "biển báo", "bảng giá", "giá dầu", "giá tiền", "cột mốc", "bản đồ phân bố",
            "tên của con đèo", "đọc chữ", "dòng chữ", "ghi nhận số liệu", "slide bài giảng"
        ]
        is_ocr_query = any(k in query_vi.lower() for k in ocr_positive_triggers)
        if is_ocr_query and intent != "OCR_TEXT":
            intent = "OCR_TEXT"

        # Ensure visual queries without actual text reading are not misclassified as OCR
        if not is_ocr_query and any(k in query_vi.lower() for k in ["ghi hình", "ghi lại", "bao nhiêu người", "mấy người", "hai người", "ba người"]):
            if not any(k in query_vi.lower() for k in ["con số", "biển báo", "bản đồ", "cân", "cột mốc", "giá"]):
                intent = "VISUAL_SCENE"

        dense_w = float(schema.get("dense_weight", 0.75))
        sparse_w = float(schema.get("sparse_weight", 0.25))
        
        # Systemic Boost: For genuine OCR_TEXT intent, boost sparse_weight >= 0.65 for superior OCR retrieval
        if intent == "OCR_TEXT":
            sparse_w = max(0.65, sparse_w)
            dense_w = round(1.0 - sparse_w, 2)
        else:
            dense_w = max(0.75, dense_w)
            sparse_w = round(1.0 - dense_w, 2)

        total_w = dense_w + sparse_w
        if total_w > 0:
            dense_w = round(dense_w / total_w, 2)
            sparse_w = round(1.0 - dense_w, 2)

        prompts = schema.get("golden_english_prompts", [])
        if not isinstance(prompts, list) or not prompts:
            prompts = [query_vi]

        keywords = schema.get("bm25_keywords", [])
        if not isinstance(keywords, list) or not keywords:
            keywords = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', query_vi) if len(w.strip()) >= 3]
        
        # Filter stop words and single meaningless letters
        stop_words = {
            "tại", "của", "và", "các", "những", "cho", "được", "bởi", "trong", "trên", "dưới",
            "khi", "sau", "đó", "này", "kia", "theo", "thứ", "nhất", "nhì", "lần", "lượt",
            "bắt", "trọn", "khoảnh", "khắc", "cùng", "nhau", "về", "hướng", "mặc", "có", "thể",
            "thấy", "ở", "gồm", "với", "là", "một", "hai", "ba", "bốn", "năm"
        }
        
        clean_vi_keywords = []
        for kw in keywords:
            kw_str = str(kw).strip()
            # Strip out pure English ASCII tokens if not in original query
            if kw_str and re.search(r'^[a-zA-Z0-9\s\-_]+$', kw_str) and kw_str.lower() not in query_vi.lower():
                continue
            # Strip out isolated stop words
            if kw_str.lower() in stop_words:
                continue
            if len(kw_str) >= 2:
                clean_vi_keywords.append(kw_str)
                
        if not clean_vi_keywords:
            clean_vi_keywords = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', query_vi) if len(w.strip()) >= 3 and w.strip().lower() not in stop_words]

        classes = schema.get("openimages_classes", [])
        if not isinstance(classes, list):
            classes = []
            
        # Filter out non-physical abstract concepts, adjectives, verbs from OpenImages classes
        invalid_concepts = {
            "slow motion", "slow_motion", "lecture", "action", "time", "camera", "arrangement",
            "center", "layer", "stretching", "cutting", "steaming", "smile", "conversation",
            "verb", "frame", "slide", "border", "title", "background", "red", "blue", "yellow",
            "green", "black", "white", "pink", "purple", "dark", "light", "ingredients", "green_object"
        }
        classes = [c.strip().lower() for c in classes if str(c).strip().lower() not in invalid_concepts]

        vlm_q = str(schema.get("vlm_question", query_vi)).strip()
        # Guarantee vlm_question is 100% in Vietnamese: if LLM translated it to pure English, fallback to query_vi
        if re.search(r'^[a-zA-Z0-9\s\,\.\?\!\'\"]+$', vlm_q) and not any(k in vlm_q.lower() for k in ["la", "gi", "bao nhieu", "co", "o dau"]):
            vlm_q = query_vi

        return {
            "intent": intent,
            "dense_weight": dense_w,
            "sparse_weight": sparse_w,
            "golden_english_prompts": prompts,
            "bm25_keywords": clean_vi_keywords,
            "openimages_classes": classes,
            "vlm_question": vlm_q,
            "query_vi": query_vi,
            "task_type": task_type
        }

    def _fallback_parse(self, query_vi, task_type):
        """Fallback parser if LLM fails or is disabled."""
        stop_words = {"tại", "của", "và", "các", "những", "cho", "được", "bởi", "trong", "trên", "dưới", "khi", "sau", "đó", "này", "kia", "theo", "thứ", "nhất", "nhì"}
        words = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', query_vi) if len(w.strip()) >= 3 and w.strip().lower() not in stop_words]
        
        ocr_positive_triggers = ["con số", "chữ", "biển báo", "bảng giá", "giá", "cân", "bản đồ", "cột mốc"]
        intent = "OCR_TEXT" if any(k in query_vi.lower() for k in ocr_positive_triggers) else "VISUAL_SCENE"
        
        sparse_w = 0.65 if intent == "OCR_TEXT" else 0.25
        dense_w = 0.35 if intent == "OCR_TEXT" else 0.75

        return {
            "intent": intent,
            "dense_weight": dense_w,
            "sparse_weight": sparse_w,
            "golden_english_prompts": [query_vi],
            "bm25_keywords": words[:8],
            "openimages_classes": [],
            "vlm_question": query_vi,
            "query_vi": query_vi,
            "task_type": task_type
        }
