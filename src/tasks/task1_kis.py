import os
import glob
import numpy as np

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """Anh xa tu chi so vector sang frame_id thuc te (doc tu anh .jpg hoac file CSV map-keyframes)."""
    if keyframes_dir and os.path.exists(keyframes_dir):
        for root, dirs, _ in os.walk(keyframes_dir):
            if video_id in dirs:
                video_folder = os.path.join(root, video_id)
                img_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
                if 0 <= frame_idx < len(img_paths):
                    return os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]
                break

    if metadata_dir and os.path.exists(metadata_dir):
        for root, _, files in os.walk(metadata_dir):
            target_csv = f"{video_id}.csv"
            if target_csv in files:
                csv_path = os.path.join(root, target_csv)
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    if 0 <= frame_idx < len(df):
                        return str(df.iloc[frame_idx, 0])
                except Exception:
                    pass

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
