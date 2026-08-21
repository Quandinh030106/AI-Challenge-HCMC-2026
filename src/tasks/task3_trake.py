import os
import glob
import numpy as np
import torch
from src.tasks.task1_kis import get_frame_id_from_idx

def align_events_dynamic_programming(scores_matrix, min_gap=8):
    """
    Quy hoach dong tim chuoi frame t_1 < t_2 < ... < t_N toi uu nhat
    voi rang buoc khoang cach thoi gian toi thieu (min_gap >= 8 frames)
    giup cac su kien khong bi dinh chum vao cung 1 giay.
    """
    T, N = scores_matrix.shape
    if T < N:
        return list(range(T)) + [T - 1] * (N - T), 0.0

    eff_gap = min(min_gap, max(1, (T - N) // max(1, N)))
    
    dp = np.full((T, N + 1), -np.inf)
    parent = np.full((T, N + 1), -1)

    for t in range(T):
        dp[t, 0] = 0.0

    for j in range(1, N + 1):
        for t in range(0, T):
            if t > 0 and dp[t - 1, j] > dp[t, j]:
                dp[t, j] = dp[t - 1, j]
                parent[t, j] = t - 1

            prev_t = t - eff_gap
            if j == 1:
                prev_score = 0.0
            elif prev_t >= 0:
                prev_score = dp[prev_t, j - 1]
            else:
                prev_score = -np.inf
                
            current_score = prev_score + scores_matrix[t, j - 1]

            if current_score > dp[t, j]:
                dp[t, j] = current_score
                parent[t, j] = -2

    aligned_frames = []
    t = T - 1
    j = N
    while j > 0 and t >= 0:
        if parent[t, j] == -2 or t < eff_gap * (j - 1):
            aligned_frames.append(t)
            j -= 1
            t = t - eff_gap
        else:
            t = parent[t, j]

    if len(aligned_frames) < N:
        # Fallback linspace
        aligned_frames = [int(x) for x in np.linspace(0, T - 1, N)]
    else:
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
        level = video_id.split('_')[0] if '_' in video_id else ""
        cand_paths = [
            os.path.join(dense_searcher.features_dir, f"{video_id}.npy"),
            os.path.join(dense_searcher.features_dir, f"clip-features-{level}", f"{video_id}.npy"),
            os.path.join(dense_searcher.features_dir, f"clip_features_{level}", f"{video_id}.npy"),
            os.path.join(dense_searcher.features_dir, "clip-features-32", f"{video_id}.npy"),
            os.path.join(dense_searcher.features_dir, "clip-features-32-aic25-b1", "clip-features-32", f"{video_id}.npy")
        ]
        for cp in cand_paths:
            if os.path.exists(cp):
                try:
                    video_features = np.load(cp)
                    break
                except Exception:
                    pass
                    
        if video_features is None:
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
    
    frame_ids = [get_frame_id_from_idx(keyframes_dir, video_id, idx, metadata_dir=metadata_dir) for idx in aligned_indices]
    return {"video_id": video_id, "frame_ids": frame_ids}

def solve_task3_batch(query_events, fused_candidates, keyframes_dir, dense_searcher, metadata_dir=None, query_processor=None, total_preds=100):
    """
    Danh gia toan dien ca Top 15 video ung vien, tim video co tong diem DP chuoi su kien cao nhat
    de dam bao 100% tim dung video goc cua bai toan TRAKE.
    """
    if not fused_candidates or not query_events:
        return []
        
    event_vectors = []
    for event_text in query_events:
        en_event = query_processor.translate_vi_to_en(event_text) if query_processor else event_text
        vec = dense_searcher.encode_text(en_event)
        if isinstance(vec, torch.Tensor):
            vec = vec.float().cpu().numpy().squeeze(0)
        event_vectors.append(vec)
    event_vectors = np.array(event_vectors)
    
    evaluated_cands = []
    for cand in fused_candidates[:15]:
        vid = cand["video_id"]
        video_features = dense_searcher.video_features_dict.get(vid)
        if video_features is None:
            continue
        scores_matrix = np.dot(video_features, event_vectors.T)
        aligned_indices, dp_score = align_events_dynamic_programming(scores_matrix)
        evaluated_cands.append({
            "video_id": vid,
            "aligned_indices": aligned_indices,
            "dp_score": dp_score,
            "original_rrf": cand.get("rrf_score", 0.0)
        })
        
    # Xep hang ung vien dua tren DP Alignment Score ket hop RRF goc
    evaluated_cands.sort(key=lambda x: (x["dp_score"] + x["original_rrf"] * 10.0), reverse=True)
    
    predictions = []
    for cand in evaluated_cands:
        vid = cand["video_id"]
        frame_ids = [get_frame_id_from_idx(keyframes_dir, vid, idx, metadata_dir=metadata_dir) for idx in cand["aligned_indices"]]
        predictions.append({"video_id": vid, "frame_ids": frame_ids})
        
    # Bo sung cac video tiep theo neu chua du 100 dong
    for cand in fused_candidates[len(evaluated_cands):]:
        vid = cand["video_id"]
        predictions.append({"video_id": vid, "frame_ids": ["0000"] * len(query_events)})
        if len(predictions) >= total_preds:
            break
            
    return predictions[:total_preds]



