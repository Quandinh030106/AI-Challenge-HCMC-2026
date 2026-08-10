import os
import glob
import numpy as np
from src.tasks.task1_kis import get_frame_id_from_idx

def align_events_dynamic_programming(scores_matrix):
    """
    Sử dụng Quy hoạch động (Dynamic Programming) để tìm chuỗi frame t_1 < t_2 < ... < t_N
    sao cho tổng điểm tương đồng của chuỗi sự kiện được cực đại hóa.
    
    scores_matrix: Ma trận có kích thước [T, N], chứa điểm cosine similarity
                   giữa T frame ảnh và N câu mô tả sự kiện con.
    """
    T, N = scores_matrix.shape
    if T < N:
        # Nếu số frame ít hơn số sự kiện, không thể sắp xếp tuyến tính
        return list(range(T)) + [T - 1] * (N - T), 0.0

    # dp[t, j] là điểm số cực đại khi căn chỉnh j sự kiện đầu tiên vào các frame từ 0 đến t
    dp = np.full((T, N + 1), -np.inf)
    # parent dùng để lưu vết đường đi phục vụ backtracking
    parent = np.full((T, N + 1), -1)

    # Khởi tạo: Căn chỉnh 0 sự kiện có điểm là 0
    for t in range(T):
        dp[t, 0] = 0.0

    for j in range(1, N + 1):
        for t in range(j - 1, T):
            # Lựa chọn 1: Không chọn frame t cho sự kiện j (sự kiện j đã được căn chỉnh vào frame trước t)
            if t > 0:
                dp[t, j] = dp[t - 1, j]
                parent[t, j] = t - 1

            # Lựa chọn 2: Chọn frame t cho sự kiện j (sự kiện j-1 phải được chọn trước frame t, tức là từ frame 0 đến t-1)
            prev_score = dp[t - 1, j - 1] if t > 0 else (0.0 if j == 1 else -np.inf)
            current_score = prev_score + scores_matrix[t, j - 1]

            if current_score > dp[t, j]:
                dp[t, j] = current_score
                parent[t, j] = -2  # Đánh dấu chuyển trạng thái chọn frame t cho sự kiện j

    # Truy vết ngược (Backtracking) để tìm danh sách các frame tối ưu
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

def solve_task3(query_events, fused_candidates, keyframes_dir, dense_searcher):
    """
    Giải quyết Task 3: Căn chỉnh chuỗi sự kiện theo thời gian (TRAKE)
    1. Nhận câu mô tả chuỗi sự kiện và mã hóa từng sự kiện con thành vector.
    2. Chọn video ứng viên hàng đầu từ kết quả Fusion.
    3. Tính toán ma trận độ tương đồng Cosine giữa các frame và chuỗi sự kiện.
    4. Áp dụng quy hoạch động để tìm và căn chỉnh chuỗi frame thỏa mãn thứ tự thời gian tăng dần.
    """
    if not fused_candidates or not query_events:
        return {"video_id": "none", "frame_ids": []}
        
    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    
    # Đọc file vector npy của video ứng viên tốt nhất
    feature_file = glob.glob(os.path.join(dense_searcher.features_dir, f"**/{video_id}.npy"), recursive=True)
    if not feature_file:
        feature_file = glob.glob(os.path.join(dense_searcher.features_dir, f"{video_id}.npy"))
        
    if not feature_file:
        return {"video_id": video_id, "frame_ids": ["0000"] * len(query_events)}
        
    try:
        video_features = np.load(feature_file[0]) # Shape: [T, D]
    except Exception as e:
        print(f"TRAKE: Lỗi load vector đặc trưng của video {video_id}: {e}")
        return {"video_id": video_id, "frame_ids": ["0000"] * len(query_events)}
        
    # Mã hóa các mô tả sự kiện con thành vector
    event_vectors = []
    for event_text in query_events:
        vector = dense_searcher.encode_text(event_text)
        event_vectors.append(vector)
    event_vectors = np.array(event_vectors) # Shape: [N, D]
    
    # Tính ma trận độ tương đồng Cosine giữa T frame và N sự kiện
    scores_matrix = np.dot(video_features, event_vectors.T) # Shape: [T, N]
    
    # Áp dụng Quy hoạch động để giải bài toán căn chỉnh tuần tự
    aligned_indices, path_score = align_events_dynamic_programming(scores_matrix)
    
    # Ánh xạ các chỉ số index tối ưu sang frame_id thực tế
    frame_ids = []
    for idx in aligned_indices:
        fid = get_frame_id_from_idx(keyframes_dir, video_id, idx)
        frame_ids.append(fid)
        
    return {
        "video_id": video_id,
        "frame_ids": frame_ids
    }
