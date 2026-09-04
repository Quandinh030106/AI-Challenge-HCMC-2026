# ==============================================================================
# AIC 2026 - TRAKE (TEMPORAL RETRIEVAL & ALIGNMENT) SOLVER VIA VITERBI DP
# ==============================================================================
import numpy as np
from typing import List, Dict, Any

class TRAKESolver:
    """
    Solves Temporal Retrieval and Alignment of Key Events (TRAKE) queries.
    Uses Viterbi Dynamic Programming to find global optimal semantic keyframes:
    t_1 < t_2 < ... < t_N strictly increasing in time.
    """
    def __init__(self, db_manager=None, clip_encoder=None):
        self.db_manager = db_manager
        self.clip_encoder = clip_encoder

    def solve(
        self,
        parsed_schema: dict,
        candidates: List[Dict[str, Any]],
        total_preds: int = 100
    ) -> List[Dict[str, Any]]:
        """Solves TRAKE query and produces 100 strictly aligned candidate sequences."""
        events = parsed_schema.get("events")
        if not events or not isinstance(events, list):
            events = parsed_schema.get("bm25_keywords", [])
        if not events:
            events = [w.strip() for w in parsed_schema.get("query_vi", "").split() if len(w.strip()) >= 3][:4]
        if not events:
            events = ["sự kiện 1", "sự kiện 2"]

        n_events = len(events)
        aligned_results = []

        if not candidates:
            candidates = [{"video_id": f"L21_V{i:03d}"} for i in range(1, total_preds + 1)]

        for cand in candidates[:total_preds]:
            vid = cand["video_id"]
            
            # Fetch video frame timeline from LanceDB
            timeline_records = []
            if self.db_manager is not None:
                timeline_records = self.db_manager.fetch_video_timeline(vid)

            seq_score = float(cand.get("score", 0.0))
            if timeline_records and len(timeline_records) >= n_events and self.clip_encoder is not None:
                # Encode events into CLIP vectors
                event_vecs = self.clip_encoder.encode_prompts(events)
                
                # Extract frame vectors from timeline records
                frame_vecs = np.array([r["vector"] for r in timeline_records], dtype=np.float32)
                
                # Compute similarity matrix: (n_events, n_frames)
                sim_matrix = np.dot(event_vecs, frame_vecs.T) # shape: (n_events, n_frames)
                
                aligned_indices, seq_score = self._align_viterbi_dp(sim_matrix)
                raw_frame_ids = [int(timeline_records[idx]["frame_id"]) for idx in aligned_indices]
            else:
                n_total = len(timeline_records) if timeline_records else 100
                aligned_indices = [int(x) for x in np.linspace(0, max(0, n_total - 1), n_events)]
                if timeline_records:
                    raw_frame_ids = [int(timeline_records[idx]["frame_id"]) for idx in aligned_indices]
                else:
                    raw_frame_ids = [idx * 30 for idx in aligned_indices]

            # Guarantee strictly monotonic increasing frame IDs: t_1 < t_2 < ... < t_N
            for i in range(1, len(raw_frame_ids)):
                if raw_frame_ids[i] <= raw_frame_ids[i - 1]:
                    raw_frame_ids[i] = raw_frame_ids[i - 1] + 10

            aligned_results.append({
                "video_id": vid,
                "frame_ids": raw_frame_ids,
                "score": float(seq_score)
            })

        aligned_results.sort(key=lambda x: x["score"], reverse=True)
        return aligned_results

    def _align_viterbi_dp(self, sim_matrix: np.ndarray) -> tuple:
        """
        Dynamic Programming Viterbi algorithm to find sequence of frame indices
        0 <= t_1 < t_2 < ... < t_N < n_frames that maximizes total alignment similarity.
        Returns (aligned_indices, average_alignment_score).
        """
        n_events, n_frames = sim_matrix.shape
        if n_frames < n_events:
            return ([int(x) for x in np.linspace(0, max(0, n_frames - 1), n_events)], 0.0)

        dp = np.full((n_events, n_frames), -np.inf, dtype=np.float32)
        parent = np.zeros((n_events, n_frames), dtype=np.int32)

        # Base case: event 0
        dp[0, :] = sim_matrix[0, :]

        # DP recurrence: for event e, previous frame must be < current frame
        for e in range(1, n_events):
            for t in range(e, n_frames):
                best_prev_t = int(np.argmax(dp[e - 1, :t]))
                dp[e, t] = dp[e - 1, best_prev_t] + sim_matrix[e, t]
                parent[e, t] = best_prev_t

        # Backtrack optimal path
        best_end_t = int(np.argmax(dp[n_events - 1, :]))
        best_total_score = float(dp[n_events - 1, best_end_t])
        aligned_idxs = [0] * n_events
        aligned_idxs[n_events - 1] = best_end_t

        curr_t = best_end_t
        for e in range(n_events - 1, 0, -1):
            curr_t = parent[e, curr_t]
            aligned_idxs[e - 1] = curr_t

        avg_score = best_total_score / max(1, n_events)
        return aligned_idxs, avg_score
