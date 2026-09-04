# ==============================================================================
# AIC 2026 - GAUSSIAN TEMPORAL SMOOTHING ALGORITHM
# ==============================================================================
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Any, Tuple

class GaussianTemporalSmoother:
    """
    Applies 1D Gaussian Kernel Convolution on video frame similarity timelines.
    Suppresses single-frame noise spikes and enhances sustained multi-second video actions.
    """
    def __init__(self, sigma: float = 1.5, window_radius: int = 4):
        self.sigma = sigma
        self.window_radius = window_radius

    def smooth_timeline(self, frame_scores: np.ndarray) -> np.ndarray:
        """Applies 1D Gaussian smoothing to a continuous 1D array of frame scores."""
        if len(frame_scores) < 3:
            return frame_scores
        return gaussian_filter1d(frame_scores.astype(np.float32), sigma=self.sigma, mode="nearest")

    def aggregate_video_candidates(
        self,
        retrieved_records: List[Dict[str, Any]],
        top_k_videos: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Groups raw retrieved frame matches by video_id, applies temporal continuity boosting,
        and selects the optimal peak semantic frame for each candidate video.
        """
        videos_dict = {}
        for rec in retrieved_records:
            vid = rec["video_id"]
            if vid not in videos_dict:
                videos_dict[vid] = []
            videos_dict[vid].append(rec)

        candidate_results = []
        for vid, frames in videos_dict.items():
            frames.sort(key=lambda x: x["frame_idx"])
            
            # Extract raw scores (1 - cosine_distance or hybrid_score)
            raw_scores = []
            for f in frames:
                score = f.get("_distance", None)
                if score is not None:
                    sim = max(0.0, 1.0 - float(score))
                else:
                    sim = float(f.get("_score", f.get("score", 0.5)))
                raw_scores.append(sim)

            raw_scores_arr = np.array(raw_scores, dtype=np.float32)
            smoothed_scores = self.smooth_timeline(raw_scores_arr)

            best_local_idx = int(np.argmax(smoothed_scores))
            best_frame_record = frames[best_local_idx]
            peak_smoothed_score = float(smoothed_scores[best_local_idx])
            
            # Action density bonus: boost videos with continuous high-scoring frame clusters
            high_density_count = np.sum(smoothed_scores > (peak_smoothed_score * 0.8))
            continuity_bonus = float(min(0.15, high_density_count * 0.02))

            final_video_score = peak_smoothed_score + continuity_bonus

            candidate_results.append({
                "video_id": vid,
                "score": final_video_score,
                "best_frame_idx": int(best_frame_record["frame_idx"]),
                "best_frame_id": int(best_frame_record["frame_id"]),
                "image_path": best_frame_record.get("image_path", ""),
                "pts_time": float(best_frame_record.get("pts_time", 0.0)),
                "detected_objects": best_frame_record.get("detected_objects", ""),
                "ocr_text": str(best_frame_record.get("ocr_text", "")),
                "keyframe_caption": str(best_frame_record.get("keyframe_caption", "")),
                "all_frame_scores": smoothed_scores
            })

        candidate_results.sort(key=lambda x: x["score"], reverse=True)
        return candidate_results[:top_k_videos]
