import os
import glob
import numpy as np

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx):
    """
    Ánh xạ từ chỉ số vector đặc trưng (0, 1, 2...) sang frame_id thực tế (ví dụ: '0000', '0005')
    bằng cách quét danh sách file ảnh thực tế của video đó.
    """
    # Tìm kiếm thư mục chứa video_id trong tất cả các folder con của keyframes_dir
    search_path = os.path.join(keyframes_dir, "**", video_id)
    video_folders = glob.glob(search_path, recursive=True)
    
    if video_folders:
        video_folder = video_folders[0]
        img_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
        if 0 <= frame_idx < len(img_paths):
            return os.path.splitext(os.path.basename(img_paths[frame_idx]))[0]
            
    # Fallback nếu không tìm thấy ảnh (giả định 1-to-1)
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
