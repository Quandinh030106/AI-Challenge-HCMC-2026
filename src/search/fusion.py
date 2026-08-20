def reciprocal_rank_fusion(dense_results, sparse_results, k=60, dense_weight=1.0, sparse_weight=1.0, dense_dict=None):
    """
    Kết hợp kết quả từ Dense Search (CLIP/SigLIP) và Sparse Search (BM25)
    sử dụng thuật toán Reciprocal Rank Fusion (RRF) có trọng số động.
    """
    rrf_scores = {}
    
    # 1. Tính toán thứ hạng và điểm RRF cho Dense Search
    for rank, item in enumerate(dense_results):
        video_id = item["video_id"]
        if video_id not in rrf_scores:
            rrf_scores[video_id] = {
                "video_id": video_id,
                "rrf_score": 0.0,
                "dense_info": item
            }
        rrf_scores[video_id]["rrf_score"] += dense_weight * (1.0 / (k + rank + 1))
        
    # 2. Tính toán thứ hạng và điểm RRF cho Sparse Search
    for rank, item in enumerate(sparse_results):
        video_id = item["video_id"]
        dense_info = None
        if dense_dict and video_id in dense_dict:
            dense_info = dense_dict[video_id]
            
        if video_id not in rrf_scores:
            rrf_scores[video_id] = {
                "video_id": video_id,
                "rrf_score": 0.0,
                "dense_info": dense_info
            }
        else:
            if rrf_scores[video_id]["dense_info"] is None and dense_info is not None:
                rrf_scores[video_id]["dense_info"] = dense_info
                
        rrf_scores[video_id]["rrf_score"] += sparse_weight * (1.0 / (k + rank + 1))
        
    # 3. Sắp xếp lại danh sách kết quả theo điểm RRF giảm dần
    fused_results = list(rrf_scores.values())
    fused_results = sorted(fused_results, key=lambda x: x["rrf_score"], reverse=True)
    return fused_results


