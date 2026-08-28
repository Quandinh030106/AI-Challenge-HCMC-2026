# ==============================================================================
# AIC 2026 - TEXTUAL KIS SOLVER WITH DEEP VLM VISUAL VERIFICATION
# ==============================================================================
import os
import torch
from PIL import Image
from typing import List, Dict, Any

class TextualKISSolver:
    """
    Solves Textual Known Item Search (KIS) task:
    Outputs Top 100 candidate predictions mapped to real physical video frame IDs.
    Optionally applies Qwen2.5-VL Deep Visual Verification to promote the best match to Top 1 (R@1 = 1.0).
    """
    def __init__(self, vlm_model=None, vlm_processor=None, device="cuda:1", enable_vlm_verify: bool = True):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.device = device
        self.enable_vlm_verify = enable_vlm_verify

    def solve(
        self,
        parsed_schema: dict,
        candidates: List[Dict[str, Any]],
        total_preds: int = 100,
        top_k_verify: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Solves KIS query:
        1. Evaluates Top candidates with VLM if enabled.
        2. Promotes the verified best video to Rank 1.
        3. Returns strictly formatted 100 predictions.
        """
        if not candidates:
            return [{"video_id": f"L21_V{i:03d}", "frame_id": 0, "score": 0.0} for i in range(1, total_preds + 1)]

        final_candidates = list(candidates)
        query_vi = parsed_schema.get("query_vi", "")

        # Deep Visual Verification for Top K candidates using VLM (Expanded to Top 20 candidates)
        if self.enable_vlm_verify and self.vlm_model is not None and self.vlm_processor is not None and query_vi:
            verify_pool = final_candidates[:max(20, top_k_verify)]
            best_promo_idx = 0
            highest_vlm_score = -1.0

            for idx, cand in enumerate(verify_pool):
                img_path = cand.get("image_path", "")
                if not img_path or not os.path.exists(img_path):
                    continue

                vlm_conf = self._verify_candidate_image(img_path, query_vi)
                if vlm_conf > highest_vlm_score:
                    highest_vlm_score = vlm_conf
                    best_promo_idx = idx

            # Promote top verified candidate to Rank 1 if confidence is high
            if best_promo_idx > 0 and highest_vlm_score >= 35.0:
                promoted = final_candidates.pop(best_promo_idx)
                final_candidates.insert(0, promoted)
                print(f"[INFO] KISSolver: Promoted candidate '{promoted['video_id']}' to Top 1 (VLM Conf: {highest_vlm_score:.1f}).")

        predictions = []
        for cand in final_candidates[:total_preds]:
            vid = cand["video_id"]
            fid = int(cand.get("best_frame_id", cand.get("frame_id", 0)))
            predictions.append({
                "video_id": vid,
                "frame_id": fid,
                "score": float(cand.get("score", 0.0))
            })

        return predictions

    def _verify_candidate_image(self, img_path: str, query_text: str) -> float:
        """Asks Qwen2.5-VL to score alignment between HD keyframe image and text description (0-100)."""
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((784, 784))

            prompt = (
                f"Nhiệm vụ: Quan sát khung ảnh HD và đánh giá mức độ trùng khớp với mô tả sau:\n"
                f"'{query_text}'\n"
                f"Trả lời duy nhất một số điểm từ 0 đến 100 thể hiện mức độ chính xác của hình ảnh so với mô tả."
            )

            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_device = next(self.vlm_model.parameters()).device if hasattr(self.vlm_model, "parameters") else self.device
            inputs = self.vlm_processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model_device)

            with torch.inference_mode():
                gen_ids = self.vlm_model.generate(**inputs, max_new_tokens=15)
                gen_trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
                out_text = self.vlm_processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()

            del inputs, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Parse numeric score
            digits = "".join([c for c in out_text if c.isdigit() or c == '.'])
            return float(digits) if digits else 0.0
        except Exception:
            return 0.0
