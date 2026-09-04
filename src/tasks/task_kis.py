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
    Applies Qwen2.5-VL Strict Discriminative Verification to eliminate false-positive
    adjacent videos and promote the true matching video to Top 1 (maximizing R@1).
    """
    def __init__(
        self,
        vlm_model=None,
        vlm_processor=None,
        device="cuda:1",
        enable_vlm_verify: bool = True,
        promotion_threshold: float = 75.0
    ):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.device = device
        self.enable_vlm_verify = enable_vlm_verify
        self.promotion_threshold = promotion_threshold

    def solve(
        self,
        parsed_schema: dict,
        candidates: List[Dict[str, Any]],
        total_preds: int = 100,
        top_k_verify: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Solves KIS query:
        1. Evaluates Top K candidates with Qwen2.5-VL using strict discriminative scoring.
        2. Re-ranks candidates scoring >= promotion_threshold (>= 75.0) to the top.
        3. Returns strictly formatted 100 predictions.
        """
        if not candidates:
            return [{"video_id": f"L21_V{i:03d}", "frame_id": 0, "score": 0.0} for i in range(1, total_preds + 1)]

        final_candidates = list(candidates)
        query_vi = parsed_schema.get("query_vi", "")

        # Deep Visual Verification for Top K candidates using VLM
        if self.enable_vlm_verify and self.vlm_model is not None and self.vlm_processor is not None and query_vi:
            verify_pool_size = min(len(final_candidates), top_k_verify)
            from src.utils.image_locator import resolve_keyframe_path

            scored_candidates = []
            for idx in range(verify_pool_size):
                cand = final_candidates[idx]
                vid = cand["video_id"]
                best_f_idx = int(cand.get("best_frame_idx", cand.get("frame_idx", 0)))
                img_path = resolve_keyframe_path(vid, best_f_idx, cand.get("image_path", ""))
                
                if not img_path or not os.path.exists(img_path):
                    cand["vlm_score"] = 0.0
                    scored_candidates.append((idx, cand, 0.0))
                    print(f"   ↳ [VLM {idx+1}/{verify_pool_size}] Video '{vid}' ➔ Missing Image (Score: 0.0)", flush=True)
                    continue

                vlm_conf = self._verify_candidate_image(img_path, query_vi)
                cand["vlm_score"] = vlm_conf
                scored_candidates.append((idx, cand, vlm_conf))
                print(f"   ↳ [VLM {idx+1}/{verify_pool_size}] Video '{vid}' (Frame {best_f_idx}) ➔ Score: {vlm_conf:.1f}/100", flush=True)

            # Identify candidates that passed strict verification threshold
            high_conf_matches = [item for item in scored_candidates if item[2] >= self.promotion_threshold]
            if high_conf_matches:
                # Sort high confidence matches by VLM score descending
                high_conf_matches.sort(key=lambda x: x[2], reverse=True)
                best_item = high_conf_matches[0]
                best_idx = best_item[0]
                best_cand = best_item[1]
                best_vlm_score = best_item[2]

                if best_idx > 0:
                    promoted = final_candidates.pop(best_idx)
                    final_candidates.insert(0, promoted)
                    print(f"   ✨ [PROMOTED] Video '{promoted['video_id']}' promoted to Top 1 (VLM Conf: {best_vlm_score:.1f} >= {self.promotion_threshold}).", flush=True)
                else:
                    print(f"   🛡️ [CONFIRMED] Video '{best_cand['video_id']}' maintained at Top 1 (VLM Conf: {best_vlm_score:.1f}).", flush=True)
            else:
                print(f"   ℹ️ [HYBRID PRESERVED] No candidate reached threshold {self.promotion_threshold}. Keeping hybrid Rank 1 ('{final_candidates[0]['video_id']}').", flush=True)

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
        """
        Asks Qwen2.5-VL for strict entity and action visual verification at HD resolution.
        Penalizes generic background matches (<= 25 pts) and rewards exact subject + action match (>= 80 pts).
        """
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((896, 896))

            prompt_directional = (
                f"Nhiệm vụ: Bạn là chuyên gia thẩm định video AIC 2026. Quan sát kỹ khung ảnh HD và đối chiếu với mô tả sau:\n"
                f"'{query_text}'\n\n"
                f"TIÊU CHUẨN CHẤM ĐIỂM NGHIÊM NGẶT (Thang điểm 0 - 100):\n"
                f"1. Nếu ảnh CHỈ CÓ CÙNG BỐI CẢNH CHUNG (ví dụ cùng là góc bếp, vườn cây, đường phố, trường học, lễ hội) nhưng THIẾU các vật thể chính hoặc SAI hành động/tương tác cụ thể trong mô tả: Chấm từ 0 đến 25 điểm (TUYỆT ĐỐI KHÔNG CHẤM CAO).\n"
                f"2. Nếu ảnh khớp một phần (có vật thể nhưng thiếu hành động hoặc góc quay phụ): Chấm từ 30 đến 55 điểm.\n"
                f"3. Nếu ảnh khớp rõ ràng, chứa ĐẦY ĐỦ 100% CÁC THỰC THỂ CỐT LÕI, ĐÚNG HÀNH ĐỘNG VÀ ĐẶC ĐIỂM CHI TIẾT: Chấm từ 80 đến 100 điểm.\n\n"
                f"Chỉ trả lời DUY NHẤT một số nguyên từ 0 đến 100."
            )

            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_directional}]}]
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_device = next(self.vlm_model.parameters()).device if hasattr(self.vlm_model, "parameters") else self.device
            inputs = self.vlm_processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model_device)

            with torch.inference_mode():
                gen_ids = self.vlm_model.generate(**inputs, max_new_tokens=20)
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
