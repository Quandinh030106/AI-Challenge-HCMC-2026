import os
import glob
import numpy as np
import torch
from src.tasks.task1_kis import get_frame_id_from_idx

def align_events_dynamic_programming(scores_matrix):
    """Quy hoach dong tim chuoi frame t_1 < t_2 < ... < t_N toi uu nhat."""
    T, N = scores_matrix.shape
    if T < N:
        return list(range(T)) + [T - 1] * (N - T), 0.0

    dp = np.full((T, N + 1), -np.inf)
    parent = np.full((T, N + 1), -1)

    for t in range(T):
        dp[t, 0] = 0.0

    for j in range(1, N + 1):
        for t in range(j - 1, T):
            if t > 0:
                dp[t, j] = dp[t - 1, j]
                parent[t, j] = t - 1

            prev_score = dp[t - 1, j - 1] if t > 0 else (0.0 if j == 1 else -np.inf)
            current_score = prev_score + scores_matrix[t, j - 1]

            if current_score > dp[t, j]:
                dp[t, j] = current_score
                parent[t, j] = -2

    aligned_frames = []
    t = T - 1
    j = N
    while j > 0 and t >= 0:
        if parent[t, j] == -2 or t == j - 1:
            aligned_frames.append(t)
            j -= 1
            t -= 1
        else:
            t = parent[t, j]

    aligned_frames.reverse()
    return aligned_frames, float(dp[T - 1, N])

def solve_task3(query_events, fused_candidates, keyframes_dir, dense_searcher, metadata_dir=None, query_processor=None):
    """Giai quyet Task 3: Can chinh chuoi su kien theo thoi gian (TRAKE)."""
    if not fused_candidates or not query_events:
        return {"video_id": "none", "frame_ids": []}
        
    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    
    video_features = dense_searcher.video_features_dict.get(video_id)
    if video_features is None:
        feature_file = glob.glob(os.path.join(dense_searcher.features_dir, f"**/{video_id}.npy"), recursive=True)
        if not feature_file:
            feature_file = glob.glob(os.path.join(dense_searcher.features_dir, f"{video_id}.npy"))
        if not feature_file:
            return {"video_id": video_id, "frame_ids": ["0000"] * len(query_events)}
        try:
            video_features = np.load(feature_file[0])
        except Exception:
            return {"video_id": video_id, "frame_ids": ["0000"] * len(query_events)}
        
    event_vectors = []
    for event_text in query_events:
        # Bắt buộc dịch sự kiện tiếng Việt sang tiếng Anh để CLIP hiểu chính xác 100%
        en_event = query_processor.translate_vi_to_en(event_text) if query_processor else event_text
        vec = dense_searcher.encode_text(en_event)
        if isinstance(vec, torch.Tensor):
            vec = vec.float().cpu().numpy().squeeze(0)
        event_vectors.append(vec)
    event_vectors = np.array(event_vectors)
    
    scores_matrix = np.dot(video_features, event_vectors.T)
    aligned_indices, _ = align_events_dynamic_programming(scores_matrix)
    
    frame_ids = [get_frame_id_from_idx(keyframes_dir, video_id, idx, metadata_dir=metadata_dir) for idx in aligned_indices]
    return {"video_id": video_id, "frame_ids": frame_ids}


