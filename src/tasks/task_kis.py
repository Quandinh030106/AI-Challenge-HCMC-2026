# ==============================================================================
# AIC 2026 - TEXTUAL KIS SOLVER WITH CHAIN-OF-THOUGHT VLM VERIFICATION
# ==============================================================================
import os
import re
import torch
from PIL import Image
from typing import List, Dict, Any

class TextualKISSolver:
    """
    Solves Textual Known Item Search (KIS) task:
    Outputs Top 100 candidate predictions mapped to real physical video frame IDs.
    Applies Qwen2.5-VL Chain-of-Thought (CoT) Visual Verification and 2-frame local
    temporal action confirmation to maximize Top-1 Exact Matches without bias.
    """
    def __init__(
        self,
        vlm_model=None,
        vlm_processor=None,
        db_manager=None,
        device="cuda:1",
        enable_vlm_verify: bool = True,
        promotion_threshold: float = 75.0
    ):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.db_manager = db_manager
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
        1. Evaluates Top K diverse candidates with Qwen2.5-VL using Chain-of-Thought checklist.
        2. Promotes candidates scoring >= promotion_threshold (>= 75.0) to Top 1.
        3. Returns strictly formatted 100 predictions.
        """
        if not candidates:
            return [{"video_id": f"L21_V{i:03d}", "frame_id": 0, "score": 0.0} for i in range(1, total_preds + 1)]

        final_candidates = list(candidates)
        query_vi = parsed_schema.get("query_vi", "")

        # Deep Visual Verification for Top K candidates using VLM
        if self.enable_vlm_verify and self.vlm_model is not None and self.vlm_processor is not None and query_vi:
            target_indices = self._build_diverse_verify_pool(final_candidates, top_k_verify=top_k_verify)
            from src.utils.image_locator import resolve_keyframe_path

            scored_candidates = []
            for step_i, cand_idx in enumerate(target_indices):
                cand = final_candidates[cand_idx]
                vid = cand["video_id"]
                best_f_idx = int(cand.get("best_frame_idx", cand.get("frame_idx", 0)))
                img_path = resolve_keyframe_path(vid, best_f_idx, cand.get("image_path", ""))
                
                if not img_path or not os.path.exists(img_path):
                    cand["vlm_score"] = 0.0
                    scored_candidates.append((cand_idx, cand, 0.0))
                    print(f"   ↳ [VLM {step_i+1}/{len(target_indices)}] Video '{vid}' (Rank #{cand_idx+1}) ➔ Missing Image (Score: 0.0)", flush=True)
                    continue

                vlm_conf = self._verify_candidate_image(img_path, query_vi)

                # Borderline check (35-74 pts): test adjacent keyframe (best_f_idx + 1) to capture full action
                if 35.0 <= vlm_conf < self.promotion_threshold and self.db_manager is not None:
                    next_frames = self.db_manager.fetch_frames_by_indices(vid, [best_f_idx + 1])
                    if next_frames:
                        next_p = resolve_keyframe_path(vid, best_f_idx + 1, next_frames[0].get("image_path", ""))
                        if next_p and os.path.exists(next_p):
                            next_conf = self._verify_candidate_image(next_p, query_vi)
                            if next_conf > vlm_conf:
                                vlm_conf = next_conf
                                cand["best_frame_id"] = int(next_frames[0].get("frame_id", cand.get("best_frame_id", 0)))

                cand["vlm_score"] = vlm_conf
                scored_candidates.append((cand_idx, cand, vlm_conf))
                print(f"   ↳ [VLM {step_i+1}/{len(target_indices)}] Video '{vid}' (Rank #{cand_idx+1}, Frame {best_f_idx}) ➔ Score: {vlm_conf:.1f}/100", flush=True)

            # Identify candidates that passed strict verification threshold
            high_conf_matches = [item for item in scored_candidates if item[2] >= self.promotion_threshold]
            if high_conf_matches:
                high_conf_matches.sort(key=lambda x: x[2], reverse=True)
                best_item = high_conf_matches[0]
                best_original_idx = best_item[0]
                best_cand = best_item[1]
                best_vlm_score = best_item[2]

                if best_original_idx > 0:
                    current_idx = final_candidates.index(best_cand)
                    promoted = final_candidates.pop(current_idx)
                    final_candidates.insert(0, promoted)
                    print(f"   ✨ [PROMOTED] Video '{promoted['video_id']}' (Rank #{best_original_idx+1}) promoted to Top 1 (VLM Conf: {best_vlm_score:.1f} >= {self.promotion_threshold}).", flush=True)
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

    def _build_diverse_verify_pool(self, candidates: List[Dict[str, Any]], top_k_verify: int = 12) -> List[int]:
        """
        Constructs a high-diversity verification pool across the top 45 candidates.
        Ensures diverse batch coverage and prevents single-cluster monopolization.
        """
        if len(candidates) <= top_k_verify:
            return list(range(len(candidates)))

        selected_indices = []
        selected_batches = {}

        # 1. Always include the top 3 candidates unconditionally
        for i in range(min(3, len(candidates))):
            selected_indices.append(i)
            b = candidates[i]["video_id"].split("_")[0] if "_" in candidates[i]["video_id"] else candidates[i]["video_id"]
            selected_batches[b] = selected_batches.get(b, 0) + 1

        # 2. Add high-potential candidates across top 45 with batch diversity cap
        search_horizon = min(45, len(candidates))
        for i in range(3, search_horizon):
            if len(selected_indices) >= top_k_verify:
                break
            vid = candidates[i]["video_id"]
            b = vid.split("_")[0] if "_" in vid else vid
            batch_count = selected_batches.get(b, 0)
            if batch_count < 4:
                selected_indices.append(i)
                selected_batches[b] = batch_count + 1

        # 3. Fill remaining slots if any
        for i in range(3, search_horizon):
            if len(selected_indices) >= top_k_verify:
                break
            if i not in selected_indices:
                selected_indices.append(i)

        return selected_indices

    def _verify_candidate_image(self, img_path: str, query_text: str) -> float:
        """
        Runs Chain-of-Thought (CoT) Visual Verification using Qwen2.5-VL.
        Forces the model to identify specific objects and actions before scoring,
        eliminating arbitrary heuristic score plateaus.
        """
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((896, 896))

            prompt_cot = (
                f"Nhiệm vụ: Bạn là chuyên gia thẩm định video AIC 2026. Quan sát kỹ khung ảnh HD và đối chiếu với mô tả sau:\n"
                f"'{query_text}'\n\n"
                f"Hãy phân tích theo 3 bước ngắn gọn:\n"
                f"1. Vật thể: Liệt kê các vật thể/chủ thể chính thấy trong ảnh.\n"
                f"2. Hành động: Mô tả ngắn hành động đang diễn ra.\n"
                f"3. Điểm số: Chấm điểm từ 0 đến 100 theo mức độ trùng khớp ĐẦY ĐỦ 100% cả vật thể và hành động (Bắt buộc ghi rõ ở cuối: Điểm: X/100)."
            )

            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_cot}]}]
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_device = next(self.vlm_model.parameters()).device if hasattr(self.vlm_model, "parameters") else self.device
            inputs = self.vlm_processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model_device)

            with torch.inference_mode():
                gen_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)
                gen_trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
                out_text = self.vlm_processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()

            del inputs, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Robust score extraction
            score_match = re.search(r'(?:score|điểm|kết quả)\s*[:\-\=]?\s*(\d{1,3})', out_text, re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))

            slash_match = re.search(r'(\d{1,3})\s*/\s*100', out_text)
            if slash_match:
                return float(slash_match.group(1))

            all_nums = re.findall(r'\b\d+\b', out_text)
            if all_nums:
                return float(all_nums[-1])
            return 0.0
        except Exception:
            return 0.0

