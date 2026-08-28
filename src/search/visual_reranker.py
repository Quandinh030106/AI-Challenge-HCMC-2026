import os
import re
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from src.utils import get_keyframe_path_by_index


DEFAULT_RERANKER_CONFIG = {
    "top_n_verify": 5,
    "retrieval_prior_weight": 0.55,
    "vlm_weight": 0.45,
    "min_swap_margin": 1.5,  # tren thang diem VLM 0-10
    "rank_decay": 60.0,
}


def _rank_score(rank, decay):
    """Cung cong thuc rank-score voi sequence_search.py / task3_trake.py."""
    if rank is None:
        return 0.0
    rank = max(1, int(rank))
    decay = max(1.0, float(decay))
    return float((decay + 1.0) / (decay + rank))


class VisualReRanker:
    """
    Module xac thuc thi giac dung Qwen-VL de cham diem lai Top ung vien.

    Prompt 9 - Phan B (comparative reranking):
    - Khong con hard-lock Rank-1 theo mot nguong VLM tuyet doi co dinh
      (5.5/10 nhu ban cu). Quyet dinh giu/doi Rank-1 dua tren SO SANH
      TUONG DOI (margin) voi candidate tot nhat trong Top-N, khong phai
      mot con so tach biet khong lien quan cac candidate khac.
    - Toan bo Top-N duoc xep lai theo combined_score (retrieval prior +
      VLM), giup cai thien ca Top-1 va Top-5 thay vi chi loc candidate xau.
    - Weights/margin dua vao config de sau nay ablation, khong hardcode.
    """

    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", config=None):
        self.model_id = model_id
        self.model = None
        self.processor = None
        cfg = dict(DEFAULT_RERANKER_CONFIG)
        cfg.update(config or {})
        self.config = cfg

    def _load_model(self):
        if self.model is None or self.processor is None:
            from src.tasks.task2_vqa import load_vlm
            self.model, self.processor = load_vlm(self.model_id)

    def find_keyframe_image(self, keyframes_dir, video_id, frame_idx):
        """Lay file anh theo keyframe/vector ordinal (khong doan tu frame_id thuc)."""
        try:
            return get_keyframe_path_by_index(keyframes_dir, video_id, int(frame_idx))
        except (IndexError, FileNotFoundError, ValueError):
            return None

    def verify_single_image(self, image_path, query_text):
        """Danh gia do trung khop cua anh tren thang diem 0-10."""
        self._load_model()
        prompt = (
            f"Hãy quan sát kỹ bức ảnh này và cho biết: Bức ảnh có chứa đúng phân cảnh được mô tả sau đây không: "
            f"'{query_text}'?\n"
            f"Đánh giá độ trùng khớp trên thang điểm từ 0 đến 10 (10 = hoàn toàn trùng khớp chi tiết, 0 = hoàn toàn không liên quan).\n"
            f"Trả về kết quả theo định dạng: Score: X (ví dụ: Score: 9.5 hoặc Score: 2)."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        ans = output_text[0].strip()
        score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", ans, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        else:
            score = 8.0 if any(w in ans.lower() for w in ["yes", "có", "đúng", "match"]) else 1.0

        return min(max(score, 0.0), 10.0)

    def get_top_peak_indices(self, dense_info, n_peaks=3):
        """Trich xuat cac dinh cuc dai dai dien phan bo tren cac phan khuc khac nhau cua video."""
        if not dense_info or "all_scores" not in dense_info:
            return [0]
        scores = dense_info["all_scores"]
        n_frames = len(scores)
        if n_frames <= n_peaks:
            return list(range(n_frames))

        from src.tasks.task1_kis import gaussian_smooth_scores
        smoothed = gaussian_smooth_scores(scores, sigma=1.5)

        peaks = []
        segment_len = n_frames // n_peaks
        for seg_i in range(n_peaks):
            start_s = seg_i * segment_len
            end_s = (seg_i + 1) * segment_len if seg_i < n_peaks - 1 else n_frames
            if start_s < end_s:
                local_peak = start_s + int(np.argmax(smoothed[start_s:end_s]))
                if local_peak not in peaks:
                    peaks.append(local_peak)

        global_peak = int(np.argmax(smoothed))
        if global_peak not in peaks:
            peaks.insert(0, global_peak)

        return sorted(peaks[:n_peaks])

    def rerank_candidates(self, fused_candidates, query_text, keyframes_dir, top_n_verify=None):
        """Xac thuc thi giac Multi-Peak cho Top-N ung vien va rerank comparative."""
        if not fused_candidates:
            return []

        top_n_verify = int(top_n_verify or self.config["top_n_verify"])
        candidates_to_verify = fused_candidates[:top_n_verify]
        remaining_candidates = fused_candidates[top_n_verify:]

        prior_weight = max(0.0, float(self.config["retrieval_prior_weight"]))
        vlm_weight = max(0.0, float(self.config["vlm_weight"]))
        weight_total = (prior_weight + vlm_weight) or 1.0
        rank_decay = float(self.config["rank_decay"])
        min_swap_margin = float(self.config["min_swap_margin"])

        verified_results = []
        for original_rank, cand in enumerate(candidates_to_verify):
            vid = cand["video_id"]
            dense_info = cand.get("dense_info")
            peak_indices = self.get_top_peak_indices(dense_info, n_peaks=3)

            frame_scores = []
            for p_idx in peak_indices:
                img_path = self.find_keyframe_image(keyframes_dir, vid, p_idx)
                if img_path and os.path.exists(img_path):
                    try:
                        frame_scores.append(self.verify_single_image(img_path, query_text))
                    except Exception as exc:
                        print(
                            "VisualReRanker: Canh bao verify that bai "
                            f"video={vid}, frame_idx={p_idx}: {exc}"
                        )

            vlm_score = max(frame_scores) if frame_scores else 5.0
            retrieval_prior = _rank_score(original_rank + 1, rank_decay)
            combined_score = float((
                prior_weight * retrieval_prior + vlm_weight * (vlm_score / 10.0)
            ) / weight_total)

            cand_copy = dict(cand)
            cand_copy["vlm_score"] = vlm_score
            cand_copy["retrieval_prior_score"] = retrieval_prior
            cand_copy["combined_score"] = combined_score
            cand_copy["original_rank"] = original_rank
            verified_results.append(cand_copy)

        if not verified_results:
            return remaining_candidates

        original_rank1 = verified_results[0]
        by_combined = sorted(verified_results, key=lambda x: x["combined_score"], reverse=True)
        best_by_combined = by_combined[0]

        # Comparative gate: chi doi Rank-1 goc khi candidate tot nhat co
        # VLM score vuot Rank-1 goc >= min_swap_margin (so sanh tuong doi,
        # KHONG phai nguong tuyet doi tach biet nhu ban cu).
        swap_margin = best_by_combined["vlm_score"] - original_rank1["vlm_score"]
        if best_by_combined is not original_rank1 and swap_margin >= min_swap_margin:
            final_top_n = by_combined
            decision = "swap_rank1"
        else:
            rest_sorted = sorted(
                (item for item in verified_results if item is not original_rank1),
                key=lambda x: x["combined_score"],
                reverse=True,
            )
            final_top_n = [original_rank1] + rest_sorted
            decision = "keep_rank1"

        print(
            "VisualReRanker: decision=%s | rank1_vlm=%.2f best_vlm=%.2f margin=%.2f (min=%.2f)"
            % (decision, original_rank1["vlm_score"], best_by_combined["vlm_score"], swap_margin, min_swap_margin)
        )

        return final_top_n + remaining_candidates