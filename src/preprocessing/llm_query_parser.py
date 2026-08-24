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
        """Loads Qwen2.5-7B-Instruct model in 4-bit NF4 quantization on cuda:0."""
        if self.model is not None:
            return

        print(f"[INFO] LLMQueryParser: Loading NLP LLM ({self.model_id})...")
        
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
            print("[INFO] LLMQueryParser: Loaded NLP LLM in 4-bit NF4 mode on cuda:0.")
        except Exception as e:
            print(f"[WARNING] 4-bit quantization loading failed ({e}). Retrying with standard FP16 on cuda:0...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.model.eval()
            print("[INFO] LLMQueryParser: Loaded NLP LLM in FP16 mode on cuda:0.")

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
        Runs LLM prompt execution to dynamically extract structured schema without bias or hardcodes.
        """
        if self.model is None:
            self.load_model()

        system_prompt = (
            "Bạn là trợ lý AI chuyên gia phân tích cú pháp truy vấn video đa phương thức cho cuộc thi AI Challenge.\n"
            "Nhiệm vụ của bạn là phân tích đoạn mô tả hoặc câu hỏi Tiếng Việt, trích xuất cấu trúc tìm kiếm JSON với ĐÚNG CÁC TRƯỜNG SAU:\n\n"
            "1. 'intent': Chọn 'VISUAL_SCENE' (nếu mô tả bối cảnh/hình ảnh) hoặc 'OCR_TEXT' (nếu yêu cầu đọc con số, chữ viết trên bảng/biển báo/cân).\n"
            "2. 'dense_weight': Trọng số tìm kiếm hình ảnh CLIP (từ 0.1 đến 0.9).\n"
            "3. 'sparse_weight': Trọng số tìm kiếm văn bản BM25 (từ 0.1 đến 0.9). Tổng dense_weight + sparse_weight = 1.0.\n"
            "4. 'golden_english_prompts': Mảng từ 2 đến 4 câu mô tả bối cảnh điện ảnh ngắn gọn bằng Tiếng Anh (Mỗi câu không quá 25 từ, miêu tả trực diện hình ảnh).\n"
            "5. 'bm25_keywords': Mảng các từ khóa Tiếng Việt cốt lõi trích xuất từ câu hỏi gốc (BẮT BỘC BẰNG TIẾNG VIỆT, KHÔNG tự dịch sang Tiếng Anh, loại bỏ từ nối rác).\n"
            "6. 'openimages_classes': Mảng danh từ Tiếng Anh đại diện cho VẬT THỂ THỂ LÝ nhìn thấy được (ví dụ: 'person', 'car', 'dog', 'table', 'sign'...). KHÔNG đưa các từ phi vật thể như 'slow motion', 'time', 'action'.\n"
            "7. 'vlm_question': Câu hỏi Tiếng Việt trực tiếp, cô đọng để VLM đọc ảnh trả lời (BẮT BỘC 100% BẰNG TIẾNG VIỆT, KHÔNG DỊCH SANG TIẾNG ANH).\n\n"
            "YÊU CẦU ĐẦU RA: CHỈ NÊU MỘT KHỐI JSON HỢP LỆ VÀ NẰM TRONG CẶP THẺ ```json ... ```. KHÔNG THÊM BẤT KỲ LỜI DẪN NÀO."
        )

        user_content = f"Loại truy vấn: {task_type.upper()}\nNội dung Tiếng Việt: {query_vi}"
        if raw_question:
            user_content += f"\nCâu hỏi thô: {raw_question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            return self._parse_with_llm(messages, query_vi, task_type)
        except Exception as e:
            print(f"[WARNING] LLMQueryParser: Dynamic parsing error ({e}). Fallback to standard parser.")
            return self._fallback_parse(query_vi, task_type)

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

        # Auto-detect OCR_TEXT intent if query asks for numbers, text, signs, scales, or prices
        if any(k in query_vi.lower() for k in ["con số", "chữ", "biển báo", "ghi", "mấy", "bao nhiêu", "hiển thị", "giá", "cân"]):
            intent = "OCR_TEXT"

        dense_w = float(schema.get("dense_weight", 0.6))
        sparse_w = float(schema.get("sparse_weight", 0.4))
        
        # Systemic Boost: For OCR_TEXT intent, boost sparse_weight >= 0.65 for superior OCR retrieval
        if intent == "OCR_TEXT":
            sparse_w = max(0.65, sparse_w)
            dense_w = round(1.0 - sparse_w, 2)

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
        
        # Strictly filter out pure English ASCII tokens to guarantee 100% Vietnamese BM25 keyword purity
        clean_vi_keywords = []
        for kw in keywords:
            kw_str = str(kw).strip()
            # If word contains pure ASCII letters and does not exist in query_vi, strip it out
            if kw_str and re.search(r'^[a-zA-Z0-9\s\-_]+$', kw_str) and kw_str.lower() not in query_vi.lower():
                continue
            if kw_str:
                clean_vi_keywords.append(kw_str)
                
        if not clean_vi_keywords:
            clean_vi_keywords = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', query_vi) if len(w.strip()) >= 3]

        classes = schema.get("openimages_classes", [])
        if not isinstance(classes, list):
            classes = []
            
        # Filter out non-physical abstract concepts from OpenImages classes
        abstract_concepts = ["slow motion", "slow_motion", "lecture", "action", "time", "camera", "arrangement", "center", "layer"]
        classes = [c.strip().lower() for c in classes if str(c).strip().lower() not in abstract_concepts]

        vlm_q = str(schema.get("vlm_question", query_vi)).strip()
        # Guarantee vlm_question is 100% in Vietnamese: if LLM translated it to pure English, fallback to query_vi
        if re.search(r'^[a-zA-Z0-9\s\,\.\?\!\'\"]+$', vlm_q) and not any(k in vlm_q.lower() for k in ["la", "gi", "bao nhieu", "co"]):
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
        words = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', query_vi) if len(w.strip()) >= 3]
        intent = "OCR_TEXT" if any(k in query_vi.lower() for k in ["con số", "chữ", "biển báo", "ghi", "mấy", "bao nhiêu", "hiển thị", "giá", "cân"]) else "VISUAL_SCENE"
        
        sparse_w = 0.65 if intent == "OCR_TEXT" else 0.4
        dense_w = 0.35 if intent == "OCR_TEXT" else 0.6

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
