import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

class LLMQueryParser:
    """
    Dynamic NLP Semantic Parser module powered by LLM.
    Extracts structured golden schema without hardcoded dictionaries or query bias.
    Optimized with 4-bit NF4 quantization on cuda:0 to prevent Kaggle multi-GPU P2P deadlocks.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", device=None):
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.llm_available = False
        self.tokenizer = None
        self.model = None
        
        self.fallback_translator_id = "facebook/nllb-200-distilled-600M"
        self.fallback_tokenizer = None
        self.fallback_model = None

    def load_model(self):
        """Loads NLP LLM on a single GPU (cuda:0) with 4-bit NF4 quantization for maximum speed and zero P2P locks."""
        if self.llm_available:
            return
            
        print(f"[INFO] LLMQueryParser: Loading NLP LLM ({self.model_id})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            
            # Cấu hình 4-bit NF4 Quantization siêu nhẹ (~4.5GB VRAM) chạy mượt trên 1 GPU cuda:0
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.model.eval()
            self.llm_available = True
            print("[INFO] LLMQueryParser: Loaded NLP LLM successfully in 4-bit NF4 mode on cuda:0.")
        except Exception as e:
            print(f"[WARNING] 4-bit quantization loading failed ({e}). Retrying with standard FP16 on cuda:0...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="cuda:0" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                self.model.eval()
                self.llm_available = True
                print("[INFO] LLMQueryParser: Loaded NLP LLM in standard FP16 mode on cuda:0.")
            except Exception as e2:
                print(f"[WARNING] LLMQueryParser: Failed to load {self.model_id} ({e2}). Switching to fallback.")
                self.llm_available = False
                self._init_fallback_translator()

    def _init_fallback_translator(self):
        """Initializes fallback NLLB translation model if primary LLM is unavailable."""
        if self.fallback_model is not None:
            return
        try:
            self.fallback_tokenizer = AutoTokenizer.from_pretrained(self.fallback_translator_id, src_lang="vie_Latn")
            self.fallback_model = AutoModelForSeq2SeqLM.from_pretrained(self.fallback_translator_id).to(self.device)
            self.fallback_model.eval()
        except Exception as e:
            print(f"[WARNING] LLMQueryParser: Fallback translator warning ({e}).")

    def unload_model(self):
        """Unloads NLP LLM from VRAM to release memory for downstream VLM tasks."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.llm_available = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] LLMQueryParser: Unloaded NLP LLM from VRAM.")

    def parse_query_dynamically(self, query_vi, task_type="kis", raw_question=""):
        """Parses query using LLM if available, otherwise uses open-vocabulary rule fallback."""
        if self.llm_available and self.model is not None:
            return self._parse_with_llm(query_vi, task_type=task_type, raw_question=raw_question)
        else:
            return self._parse_with_fallback(query_vi, task_type=task_type, raw_question=raw_question)

    def _parse_with_llm(self, query_vi, task_type="kis", raw_question=""):
        """Extracts dynamic visual semantic schema using Qwen2.5-7B-Instruct."""
        system_prompt = (
            "Bạn là chuyên gia phân tích ngữ nghĩa thị giác đa phương thức.\n"
            "Nhiệm vụ: Hãy đọc hiểu câu hỏi Tiếng Việt được cung cấp và trích xuất ra duy nhất một đối tượng JSON chuẩn gồm các trường:\n"
            "1. 'intent': Trả về 'OCR_TEXT' nếu câu hỏi yêu cầu đọc ký tự, chữ viết, con số, văn bản, thông số hiển thị hoặc tên ghi trên đối tượng; trả về 'VISUAL_SCENE' nếu tập trung vào mô tả bối cảnh, hành động hoặc con người.\n"
            "2. 'dense_weight': Số thực từ 0.1 đến 0.9 thể hiện trọng số ưu tiên tìm kiếm hình ảnh mảng khối (CLIP dense search).\n"
            "3. 'sparse_weight': Số thực từ 0.1 đến 0.9 thể hiện trọng số ưu tiên tìm kiếm từ khóa văn bản (BM25 sparse search). Lưu ý: dense_weight + sparse_weight = 1.0.\n"
            "4. 'golden_english_prompts': Mảng từ 3 đến 5 câu Tiếng Anh tả bối cảnh thị giác điện ảnh chuẩn xác (dưới 70 từ mỗi câu, tập trung vào hình thái vật thể, thuộc tính màu sắc, góc máy và hành động chính).\n"
            "5. 'bm25_keywords': Mảng các cụm từ Tiếng Việt cốt lõi trích xuất từ câu hỏi (loại bỏ từ nối không mang hàm lượng thông tin).\n"
            "6. 'openimages_classes': Mảng các danh từ Tiếng Anh đơn (Single-word English Nouns) đại diện cho các lớp vật thể chính xuất hiện trong cảnh (theo chuẩn danh mục Google OpenImages).\n"
            "7. 'vlm_question': Câu hỏi Tiếng Việt trực tiếp, cô đọng để mô hình thị giác đọc ảnh trả lời (loại bỏ toàn bộ câu dẫn rườm rà).\n\n"
            "Yêu cầu: Xuất duy nhất mã JSON hợp lệ, không thêm bất kỳ văn bản giải thích hoặc ký tự ngoài JSON."
        )
        
        user_input = f"Loại bài toán: {task_type.upper()}\nNội dung câu hỏi Tiếng Việt: '{query_vi}'"
        if raw_question:
            user_input += f"\nCâu hỏi chi tiết: '{raw_question}'"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        try:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer([prompt_text], return_tensors="pt").to(target_device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1)
                generated_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], outputs)]
                response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                return self._normalize_schema(parsed_json, query_vi)
        except Exception as e:
            print(f"[WARNING] LLMQueryParser: Failed to generate JSON with LLM ({e}). Switching to fallback.")
            
        return self._parse_with_fallback(query_vi, task_type=task_type, raw_question=raw_question)

    def _parse_with_fallback(self, query_vi, task_type="kis", raw_question=""):
        """Open-vocabulary rule-based parsing fallback mechanism."""
        translated_en = self._translate_vi_to_en(query_vi)
        query_lower = query_vi.lower()
        
        is_ocr = (
            "?" in query_vi or 
            task_type == "qa" or 
            bool(re.search(r'\d+', query_vi)) or
            any(w in query_lower for w in ["bao nhiêu", "mấy", "gì", "nào", "chữ", "số", "bảng", "tên"])
        )
        intent = "OCR_TEXT" if is_ocr else "VISUAL_SCENE"
        
        raw_words = [w.strip() for w in re.split(r'[,.\s\?\!\:\;\-\"\']+', query_vi) if len(w.strip()) >= 2]
        generic_stopwords = {"và", "hoặc", "nhưng", "của", "cho", "trong", "trên", "dưới", "các", "những", "một", "này", "đó", "khi", "được", "bởi", "với"}
        bm25_kws = [w for w in raw_words if w.lower() not in generic_stopwords][:12]
        
        clean_en = translated_en.rstrip('.')
        golden_prompts = [
            clean_en,
            f"a photo of {clean_en}",
            f"a video scene showing {clean_en}",
            f"a high quality shot of {clean_en}"
        ]
        
        en_words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_en)
        en_stopwords = {"the", "and", "is", "are", "with", "this", "that", "from", "show", "showing", "view", "scene", "photo", "shot"}
        open_classes = [w.capitalize() for w in en_words if w.lower() not in en_stopwords][:5]
        if not open_classes:
            open_classes = ["Object"]
            
        clean_q = raw_question if raw_question else query_vi
        clean_q = re.sub(r'^(câu hỏi|hỏi|cho biết)\s*[:\.]?\s*', '', clean_q, flags=re.IGNORECASE).strip()
        
        d_weight = 0.35 if intent == "OCR_TEXT" else 0.75
        s_weight = 0.65 if intent == "OCR_TEXT" else 0.25
        
        return {
            "intent": intent,
            "dense_weight": d_weight,
            "sparse_weight": s_weight,
            "golden_english_prompts": golden_prompts,
            "bm25_keywords": bm25_kws,
            "openimages_classes": open_classes,
            "vlm_question": clean_q,
            "query_vi": query_vi,
            "query_en": translated_en
        }

    def _translate_vi_to_en(self, text_vi):
        """Translates Vietnamese query to English using NLLB fallback model."""
        if not text_vi:
            return ""
        if self.fallback_model is not None and self.fallback_tokenizer is not None:
            try:
                inputs = self.fallback_tokenizer(text_vi, return_tensors="pt").to(self.device)
                eng_token_id = self.fallback_tokenizer.convert_tokens_to_ids("eng_Latn")
                with torch.no_grad():
                    gen = self.fallback_model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=150)
                return self.fallback_tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            except Exception:
                pass
        return text_vi

    def _normalize_schema(self, parsed_json, query_vi):
        """Normalizes extracted LLM JSON object and validates fusion weight constraint."""
        intent = parsed_json.get("intent", "VISUAL_SCENE")
        
        d_w = parsed_json.get("dense_weight")
        s_w = parsed_json.get("sparse_weight")
        
        if d_w is None or s_w is None:
            d_w = 0.35 if intent == "OCR_TEXT" else 0.75
            s_w = 0.65 if intent == "OCR_TEXT" else 0.25
        else:
            try:
                d_w = float(d_w)
                s_w = float(s_w)
                total = d_w + s_w
                if total > 0:
                    d_w = round(d_w / total, 2)
                    s_w = round(1.0 - d_w, 2)
                else:
                    d_w, s_w = 0.75, 0.25
            except Exception:
                d_w, s_w = 0.75, 0.25
                
        return {
            "intent": intent,
            "dense_weight": d_w,
            "sparse_weight": s_w,
            "golden_english_prompts": parsed_json.get("golden_english_prompts", [query_vi]),
            "bm25_keywords": parsed_json.get("bm25_keywords", []),
            "openimages_classes": parsed_json.get("openimages_classes", ["Object"]),
            "vlm_question": parsed_json.get("vlm_question", query_vi),
            "query_vi": query_vi,
            "query_en": parsed_json.get("golden_english_prompts", [query_vi])[0] if parsed_json.get("golden_english_prompts") else query_vi
        }
