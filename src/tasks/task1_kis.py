import os
import glob
import numpy as np

_video_folder_cache = {}
_csv_map_cache = {}

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """
    Anh xa tu chi so vector (0, 1, 2...) sang Frame ID thoi gian thuc cua video goc (vi du: 1200, 25300).
    UU TIEN SO 1: Doc tu file CSV map-keyframes cua BTC (chua Frame ID thoi gian thuc cua video).
    FALLBACK SO 2: Neu khong co CSV map-keyframes, moi lay ten file anh keyframe.
    """
    global _video_folder_cache, _csv_map_cache
    
    # 1. UU TIEN SO 1: Tra cuu tu file CSV Mapping (map-keyframes/*.csv)
    if video_id in _csv_map_cache:
        df_col = _csv_map_cache[video_id]
        if 0 <= frame_idx < len(df_col):
            return str(df_col[frame_idx])
    elif metadata_dir and os.path.exists(metadata_dir):
        level = video_id.split('_')[0] if '_' in video_id else ""
        candidate_csvs = [
            os.path.join(metadata_dir, f"{video_id}.csv"),
            os.path.join(metadata_dir, "map-keyframes", f"{video_id}.csv"),
            os.path.join(metadata_dir, f"map-keyframes-{level}", f"{video_id}.csv"),
            os.path.join(os.path.dirname(metadata_dir), "map-keyframes-aic25-b1", "map-keyframes", f"{video_id}.csv"),
            os.path.join(os.path.dirname(metadata_dir), "map-keyframes", f"{video_id}.csv")
        ]
        
        target_csv_path = None
        for c_path in candidate_csvs:
            if os.path.exists(c_path):
                target_csv_path = c_path
                break
                
        if not target_csv_path:
            for root, _, files in os.walk(metadata_dir):
                if f"{video_id}.csv" in files:
                    target_csv_path = os.path.join(root, f"{video_id}.csv")
                    break
                    
        if target_csv_path:
            try:
                import pandas as pd
                df = pd.read_csv(target_csv_path)
                # Lay cot frame_idx hoac cot cuoi cung/dau tien chua so frame thoi gian thuc
                col_name = None
                for c in ["frame_idx", "frame_id", "frame", "frameIdx", "pts_frame"]:
                    if c in df.columns:
                        col_name = c
                        break
                if col_name:
                    values = df[col_name].tolist()
                else:
                    # Neu khong co ten cot chuan, lay cot co gia tri so lon nhat (thuong la frame_idx)
                    values = df.iloc[:, 0].tolist()
                    
                _csv_map_cache[video_id] = values
                if 0 <= frame_idx < len(values):
                    return str(values[frame_idx])
            except Exception:
                pass

    # 2. FALLBACK SO 2: Neu khong co file CSV map-keyframes, moi lay ten file anh (.jpg)
    if video_id in _video_folder_cache:
        img_paths = _video_folder_cache[video_id]
        if 0 <= frame_idx < len(img_paths):
            return os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]
    elif keyframes_dir and os.path.exists(keyframes_dir):
        level = video_id.split('_')[0] if '_' in video_id else ""
        candidate_dirs = [
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id),
            os.path.join(keyframes_dir, f"Keyframes_{level}", video_id),
            os.path.join(keyframes_dir, level, "keyframes", video_id),
            os.path.join(keyframes_dir, "keyframes", video_id),
            os.path.join(keyframes_dir, video_id)
        ]
        
        video_folder = None
        for cand in candidate_dirs:
            if os.path.exists(cand):
                video_folder = cand
                break
                
        if not video_folder:
            for root, dirs, _ in os.walk(keyframes_dir):
                if video_id in dirs:
                    video_folder = os.path.join(root, video_id)
                    break
                    
        if video_folder:
            img_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
            _video_folder_cache[video_id] = img_paths
            if 0 <= frame_idx < len(img_paths):
                return os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]

    return f"{frame_idx:04d}"


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
    """Phan bo 100 cau tra loi trai dai qua nhieu video ung vien de toi uu R@k."""
    predictions = []
    
    for rank, cand in enumerate(fused_candidates):
        video_id = cand["video_id"]
        dense_info = cand.get("dense_info")
        
        n_peaks = 4 if rank < 2 else (3 if rank < 10 else (2 if rank < 30 else 1))
            
        if dense_info is not None and "all_scores" in dense_info:
            scores = dense_info["all_scores"]
            smoothed = gaussian_smooth_scores(scores, sigma=1.5)
            top_frame_idxs = np.argsort(smoothed)[::-1][:n_peaks]
            for f_idx in top_frame_idxs:
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
        predictions.append({"video_id": "none", "frame_id": "0000"})
        
    return predictions[:total_preds]
