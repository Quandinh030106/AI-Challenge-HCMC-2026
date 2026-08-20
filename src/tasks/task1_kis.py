import os
import glob
import numpy as np

_video_folder_cache = {}
_csv_map_cache = {}

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """Anh xa chi so vector sang Frame ID thoi gian thuc cua video goc."""
    global _video_folder_cache, _csv_map_cache
    
    if video_id in _csv_map_cache:
        df_col = _csv_map_cache[video_id]
        if 0 <= frame_idx < len(df_col):
            return str(int(df_col[frame_idx]))
            
    level = video_id.split('_')[0] if '_' in video_id else ""
    candidate_csvs = []
    
    if metadata_dir:
        candidate_csvs.extend([
            os.path.join(metadata_dir, f"{video_id}.csv"),
            os.path.join(metadata_dir, "map-keyframes", f"{video_id}.csv"),
            os.path.join(metadata_dir, f"map-keyframes-{level}", f"{video_id}.csv"),
            os.path.join(metadata_dir, "map-keyframes-aic25-b1", "map-keyframes", f"{video_id}.csv"),
            os.path.join(os.path.dirname(metadata_dir), "map-keyframes-aic25-b1", "map-keyframes", f"{video_id}.csv"),
            os.path.join(os.path.dirname(metadata_dir), "map-keyframes", f"{video_id}.csv"),
            os.path.join(os.path.dirname(metadata_dir), f"{video_id}.csv")
        ])
        
    if os.path.exists("/kaggle/input"):
        candidate_csvs.extend([
            f"/kaggle/input/ai-challenge-hcmc-2026-metadata/map-keyframes-aic25-b1/map-keyframes/{video_id}.csv",
            f"/kaggle/input/ai-challenge-hcmc-2026-metadata/map-keyframes/{video_id}.csv",
            f"/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-metadata/map-keyframes-aic25-b1/map-keyframes/{video_id}.csv",
            f"/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-metadata/map-keyframes/{video_id}.csv"
        ])
    
    target_csv_path = None
    for c_path in candidate_csvs:
        if os.path.exists(c_path):
            target_csv_path = c_path
            break

                
    if target_csv_path:
        try:
            import pandas as pd
            df = pd.read_csv(target_csv_path)
            col_name = None
            for c in df.columns:
                if any(k in str(c).lower() for k in ["frame_idx", "frame_id", "frame", "frameidx", "pts_frame"]):
                    col_name = c
                    break
            if col_name:
                values = df[col_name].tolist()
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    best_col = max(numeric_cols, key=lambda c: df[c].max())
                    values = df[best_col].tolist()
                else:
                    values = df.iloc[:, 0].tolist()
                
            _csv_map_cache[video_id] = values
            if 0 <= frame_idx < len(values):
                return str(int(values[frame_idx]))
        except Exception:
            pass

    real_time_estimate = int(frame_idx * 25) if frame_idx > 0 else 1
    return str(real_time_estimate)

def gaussian_smooth_scores(scores, sigma=1.5):
    """Lam min chuoi diem thoi gian bang Gaussian Kernel."""
    if len(scores) < 3:
        return scores
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    return np.convolve(scores, kernel, mode='same')

def solve_task1(query_text, fused_candidates, keyframes_dir, metadata_dir=None, sigma=1.5):
    """Giai quyet Task 1 (Textual KIS)."""
    if not fused_candidates:
        return {"video_id": "none", "frame_id": "0000", "score": 0.0}
        
    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    dense_info = best_candidate.get("dense_info")
    
    if dense_info is None or "all_scores" not in dense_info:
        return {"video_id": video_id, "frame_id": "0000", "score": best_candidate.get("rrf_score", 0.0)}
        
    scores = dense_info["all_scores"]
    smoothed_scores = gaussian_smooth_scores(scores, sigma=sigma)
    best_frame_idx = int(np.argmax(smoothed_scores))
    frame_id = get_frame_id_from_idx(keyframes_dir, video_id, best_frame_idx, metadata_dir=metadata_dir)
    
    return {"video_id": video_id, "frame_id": frame_id, "score": float(smoothed_scores[best_frame_idx])}

def generate_diversity_top100_kis(fused_candidates, keyframes_dir, metadata_dir=None, total_preds=100):
    """Phan bo 100 cau tra loi thong minh bang NMS va mo rong cua so thoi gian."""
    predictions = []
    
    for rank, cand in enumerate(fused_candidates):
        video_id = cand["video_id"]
        dense_info = cand.get("dense_info")
        
        if dense_info is not None and "all_scores" in dense_info:
            scores = dense_info["all_scores"]
            n_frames = len(scores)
            smoothed = gaussian_smooth_scores(scores, sigma=1.5)
            
            sorted_indices = np.argsort(smoothed)[::-1]
            selected_peaks = []
            min_distance = 8
            
            for idx in sorted_indices:
                if all(abs(idx - p) >= min_distance for p in selected_peaks):
                    selected_peaks.append(int(idx))
                if len(selected_peaks) >= 4:
                    break
                    
            frame_indices_to_take = []
            if rank == 0:
                for p in selected_peaks[:3]:
                    for delta in [0, -1, 1, 2, -2]:
                        target_f = p + delta
                        if 0 <= target_f < n_frames and target_f not in frame_indices_to_take:
                            frame_indices_to_take.append(target_f)
            elif rank < 4:
                for p in selected_peaks[:2]:
                    for delta in [0, 1, -1]:
                        target_f = p + delta
                        if 0 <= target_f < n_frames and target_f not in frame_indices_to_take:
                            frame_indices_to_take.append(target_f)
            elif rank < 12:
                frame_indices_to_take = selected_peaks[:2]
            else:
                frame_indices_to_take = selected_peaks[:1] if selected_peaks else [int(sorted_indices[0])]
                
            for f_idx in frame_indices_to_take:
                fid = get_frame_id_from_idx(keyframes_dir, video_id, int(f_idx), metadata_dir=metadata_dir)
                predictions.append({"video_id": video_id, "frame_id": fid})
                if len(predictions) >= total_preds:
                    break
        else:
            fid = get_frame_id_from_idx(keyframes_dir, video_id, 0, metadata_dir=metadata_dir)
            predictions.append({"video_id": video_id, "frame_id": fid})
            
        if len(predictions) >= total_preds:
            break
            
    while len(predictions) < total_preds:
        last_vid = fused_candidates[0]["video_id"] if fused_candidates else "none"
        predictions.append({"video_id": last_vid, "frame_id": "0000"})
        
    return predictions[:total_preds]
