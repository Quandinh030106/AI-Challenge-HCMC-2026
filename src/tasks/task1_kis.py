import os
import glob
import numpy as np

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """
    Ánh xạ từ chỉ số vector đặc trưng (0, 1, 2...) sang frame_id thực tế
    bằng cách quét ảnh keyframe thực tế hoặc file CSV map-keyframes từ BTC.
    """
    # 1. Thử quét thư mục chứa ảnh keyframe thực tế
    if keyframes_dir and os.path.exists(keyframes_dir):
        for root, dirs, _ in os.walk(keyframes_dir):
            if video_id in dirs:
                video_folder = os.path.join(root, video_id)
                img_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
                if 0 <= frame_idx < len(img_paths):
                    return os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]
                break

    # 2. Nếu không tìm thấy ảnh, thử đọc file CSV mapping (ví dụ L21_V001.csv trong map-keyframes)
    if metadata_dir and os.path.exists(metadata_dir):
        for root, _, files in os.walk(metadata_dir):
            target_csv = f"{video_id}.csv"
            if target_csv in files:
                csv_path = os.path.join(root, target_csv)
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    if 0 <= frame_idx < len(df):
                        # Lấy giá trị ở cột đầu tiên làm frame_id
                        first_col_val = df.iloc[frame_idx, 0]
                        return str(first_col_val)
                except Exception:
                    pass

    # Fallback nếu không tìm thấy dữ liệu
    return f"{frame_idx:04d}"



def solve_task1(query_text, fused_candidates, keyframes_dir, window_size=5):
    """
    Giải quyết Task 1: Tìm kiếm chính xác (KIS)
    1. Nhận video ứng viên hàng đầu từ kết quả Fusion.
    2. Áp dụng bộ lọc làm mịn (Rolling Average) trên các frame để khử nhiễu.
    3. Trả về video_id và frame_id của điểm tương đồng cao nhất.
    """
    if not fused_candidates:
        return {"video_id": "none", "frame_id": "0000", "score": 0.0}
        
    # Chọn ứng viên tốt nhất từ kết quả Fusion
    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    dense_info = best_candidate["dense_info"]
    
    if dense_info is None or "all_scores" not in dense_info:
        # Nếu chỉ tìm thấy qua BM25 không có thông tin vector đặc trưng
        return {
            "video_id": video_id,
            "frame_id": "0000",
            "score": best_candidate.get("rrf_score", 0.0)
        }
        
    scores = dense_info["all_scores"]
    
    # Áp dụng bộ lọc làm mịn (Moving Average)
    if len(scores) >= window_size:
        smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='same')
    else:
        smoothed_scores = scores
        
    # Tìm chỉ số frame có điểm cao nhất sau khi lọc
    best_frame_idx = int(np.argmax(smoothed_scores))
    best_score = float(smoothed_scores[best_frame_idx])
    
    # Ánh xạ chỉ số sang frame_id thực tế (ví dụ: '0150')
    frame_id = get_frame_id_from_idx(keyframes_dir, video_id, best_frame_idx)
    
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "score": best_score
    }
