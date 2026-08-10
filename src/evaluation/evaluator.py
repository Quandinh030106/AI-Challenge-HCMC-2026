import numpy as np

class Evaluator:
    def __init__(self, ground_truth):
        """
        ground_truth: Dict đọc từ file local_val_gt.json
        """
        # Chuyển đổi danh sách Ground Truth thành từ điển tra cứu nhanh theo query_id
        self.gt = {}
        for task in ["task1", "task2", "task3"]:
            self.gt[task] = {}
            if task in ground_truth:
                # Hỗ trợ cả trường hợp Ground Truth là List hoặc Dict
                if isinstance(ground_truth[task], list):
                    for item in ground_truth[task]:
                        self.gt[task][item["query_id"]] = item
                elif isinstance(ground_truth[task], dict):
                    self.gt[task] = ground_truth[task]
        
    def check_answer_match(self, pred, gt):
        """
        Kiểm tra độ trùng khớp câu trả lời Q&A một cách mềm dẻo (Normalized Semantic Match).
        """
        if not pred or not gt:
            return False
        pred = str(pred).lower().strip()
        gt = str(gt).lower().strip()
        
        # Khớp tuyệt đối
        if pred == gt:
            return True
        # Khớp chuỗi con
        if gt in pred or pred in gt:
            return True
        # Khớp các từ khóa cốt lõi (Giao của tập hợp từ)
        pred_words = set(pred.split())
        gt_words = set(gt.split())
        if len(pred_words.intersection(gt_words)) > 0:
            return True
        return False

    def evaluate_query(self, query_id, task_type, predictions):
        """
        Đánh giá 1 câu query duy nhất.
        predictions: Danh sách tối đa 100 câu trả lời nộp lên, sắp xếp thứ tự ưu tiên giảm dần.
        """
        if query_id not in self.gt.get(task_type, {}):
            return 0.0, [0.0] * 5
            
        gt_item = self.gt[task_type][query_id]
        gt_video = gt_item["video_id"]
        
        # 1. Tính R-Score cho từng dự đoán thứ i trong danh sách nộp (tối đa 100)
        r_scores = []
        for pred in predictions:
            pred_video = pred.get("video_id")
            
            # Sai video nhận ngay 0 điểm
            if pred_video != gt_video:
                r_scores.append(0.0)
                continue
                
            if task_type == "task1":
                # KIS: Đúng video và frame nằm trong khoảng cho phép [s, e]
                try:
                    pred_frame = int(pred.get("frame_id", -1))
                    frame_start = int(gt_item["frame_start"])
                    frame_end = int(gt_item["frame_end"])
                    if frame_start <= pred_frame <= frame_end:
                        r_scores.append(1.0)
                    else:
                        r_scores.append(0.0)
                except ValueError:
                    r_scores.append(0.0)
                    
            elif task_type == "task2":
                # Q&A: Đúng video, đúng frame và đúng câu trả lời phụ
                try:
                    pred_frame = int(pred.get("frame_id", -1))
                    frame_start = int(gt_item["frame_start"])
                    frame_end = int(gt_item["frame_end"])
                    pred_ans = pred.get("answer", "")
                    gt_ans = gt_item["answer"]
                    
                    if (frame_start <= pred_frame <= frame_end) and self.check_answer_match(pred_ans, gt_ans):
                        r_scores.append(1.0)
                    else:
                        r_scores.append(0.0)
                except ValueError:
                    r_scores.append(0.0)
                    
            elif task_type == "task3":
                # TRAKE: Tỷ lệ khớp khung hình của chuỗi sự kiện con
                events = gt_item["events"]
                pred_frames = pred.get("frame_ids", [])
                
                if len(pred_frames) != len(events):
                    r_scores.append(0.0)
                    continue
                    
                match_count = 0
                for idx, event in enumerate(events):
                    try:
                        pf = int(pred_frames[idx])
                        fs = int(event["frame_start"])
                        fe = int(event["frame_end"])
                        if fs <= pf <= fe:
                            match_count += 1
                    except ValueError:
                        pass
                # R-Score = tỷ lệ số event khớp trên tổng số event
                r_scores.append(match_count / len(events))
                
        # Điền thêm điểm 0.0 cho đủ 100 câu trả lời
        if not r_scores:
            r_scores = [0.0]
        while len(r_scores) < 100:
            r_scores.append(0.0)
            
        # 2. Tính R@k với k thuộc {1, 5, 20, 50, 100}
        k_values = [1, 5, 20, 50, 100]
        r_at_k = []
        for k in k_values:
            val = float(max(r_scores[:k]))
            r_at_k.append(val)
            
        # 3. Điểm Final Score của query là trung bình cộng của 5 mốc R@k
        final_score = float(np.mean(r_at_k))
        return final_score, r_at_k

    def evaluate_all(self, predictions_dict):
        """
        Tính điểm trung bình toàn bộ tất cả các Task.
        predictions_dict: Kết quả chạy của cả nhóm:
        {
          "task1": { "query_id_1": [pred1, pred2, ...], ... },
          "task2": { ... },
          "task3": { ... }
        }
        """
        all_final_scores = []
        r_at_k_accum = np.zeros(5)
        count = 0
        
        for task_type in ["task1", "task2", "task3"]:
            task_gt = self.gt.get(task_type, {})
            task_preds = predictions_dict.get(task_type, {})
            
            for query_id in task_gt.keys():
                preds = task_preds.get(query_id, [])
                final_score, r_at_k = self.evaluate_query(query_id, task_type, preds)
                all_final_scores.append(final_score)
                r_at_k_accum += np.array(r_at_k)
                count += 1
                
        if count == 0:
            return {"Final_Score": 0.0, "R@1": 0.0, "R@5": 0.0, "R@20": 0.0, "R@50": 0.0, "R@100": 0.0}
            
        r_at_k_avg = r_at_k_accum / count
        mean_final_score = float(np.mean(all_final_scores))
        
        return {
            "Final_Score": mean_final_score,
            "R@1": float(r_at_k_avg[0]),
            "R@5": float(r_at_k_avg[1]),
            "R@20": float(r_at_k_avg[2]),
            "R@50": float(r_at_k_avg[3]),
            "R@100": float(r_at_k_avg[4])
        }
