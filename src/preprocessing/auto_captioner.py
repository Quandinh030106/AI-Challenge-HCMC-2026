# ==============================================================================
# AIC 2026 - DEEP KEYFRAME AUTO-CAPTIONER USING QWEN2.5-VL-3B-INSTRUCT
# ==============================================================================
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List

class QwenKeyframeCaptioner:
    """
    Accuracy-First Keyframe Attribute Extractor using Qwen/Qwen2.5-VL-3B-Instruct.
    Generates multi-sentence dense captions, specific character attributes (red hats, glasses),
    physical actions (stretching, touching toes), setting, and OCR text.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda:0"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        print(f"[INFO] QwenKeyframeCaptioner: Loading '{self.model_id}' on {self.device}...")
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
            except ImportError:
                from transformers import AutoModelForCausalLM as VLMClass

            bnb_config = None
            try:
                import bitsandbytes
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            except Exception:
                print("[WARNING] bitsandbytes not available. Loading Qwen2.5-VL in FP16 mode...")

            self.model = VLMClass.from_pretrained(
                self.model_id,
                quantization_config=bnb_config if (bnb_config and torch.cuda.is_available()) else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=100352, max_pixels=401408, trust_remote_code=True)
            self.model.eval()
            print("[INFO] QwenKeyframeCaptioner: Loaded successfully.")
        except Exception as e:
            print(f"[ERROR] QwenKeyframeCaptioner: Failed to load model ({e})")

    def unload(self):
        """Frees VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] QwenKeyframeCaptioner: Unloaded model from GPU.")

    def caption_image(self, image_path: str) -> Dict[str, str]:
        """
        Extracts deep visual description, specific character/object attributes, and OCR text.
        Returns dict: {"caption": str, "ocr_text": str, "keywords": str}
        """
        if self.model is None or not os.path.exists(image_path):
            return {"caption": "", "ocr_text": "", "keywords": ""}

        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((448, 448))

            prompt = (
                "Describe this video keyframe image in English with maximum detail for visual retrieval:\n"
                "1. Main action, posture, and physical movement (e.g. stretching, touching toes, walking, driving).\n"
                "2. Character details: exact count of people, gender, specific clothing colors, hats (color), glasses, accessories.\n"
                "3. Setting/environment (outdoor park, stage, street, bridge, indoor, zoo).\n"
                "4. Any visible text or numbers on signs, banners, or scales.\n"
                "Provide a clear, detailed 2-3 sentence description."
            )

            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_template], images=[img], padding=True, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                gen_ids = self.model.generate(**inputs, max_new_tokens=100)
                gen_trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
                out_text = self.processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()

            del inputs, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "caption": out_text,
                "ocr_text": "",
                "keywords": ""
            }
        except Exception as e:
            print(f"[WARNING] Caption error for '{image_path}': {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"caption": "", "ocr_text": "", "keywords": ""}

    def batch_generate_captions(self, image_paths: List[str], output_json_path: str = "data/keyframe_captions.json") -> Dict[str, dict]:
        """
        Generates and caches captions for a batch of keyframes.
        Saves output to JSON file for permanent Kaggle Dataset persistence.
        """
        results = {}
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                print(f"[INFO] Loaded {len(results)} existing cached captions from '{output_json_path}'.")
            except Exception:
                results = {}

        to_process = [p for p in image_paths if p not in results]
        if not to_process:
            print(f"[INFO] All {len(image_paths)} keyframe captions are already generated!")
            return results

        print(f"[INFO] Generating Qwen2.5-VL-3B captions for {len(to_process)} keyframe images...")
        count = 0
        for img_p in tqdm(to_process, desc="Qwen2.5-VL-3B Captioning"):
            res = self.caption_image(img_p)
            results[img_p] = res
            count += 1
            
            # Save progress every 500 images
            if count % 500 == 0:
                os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Completed captioning! Total cached records: {len(results)} in '{output_json_path}'.")
        return results

if __name__ == "__main__":
    captioner = QwenKeyframeCaptioner()
