import numpy as np

class Evaluator:
    def __init__(self, ground_truth):
        """Khoi tao bo danh gia tu ground_truth dict."""
        self.gt = {}
        for task in ["task1", "task2", "task3"]:
            self.gt[task] = {}
            if task in ground_truth:
                if isinstance(ground_truth[task], list):
                    for item in ground_truth[task]:
                        self.gt[task][item["query_id"]] = item
                elif isinstance(ground_truth[task], dict):
                    self.gt[task] = ground_truth[task]
        
    def check_answer_match(self, pred, gt):
        """Kiem tra do khop cau tra loi Q&A."""
        if not pred or not gt:
            return False
        pred_str = str(pred).lower().strip()
        gt_str = str(gt).lower().strip()
        
        if pred_str == gt_str or gt_str in pred_str or pred_str in gt_str:
            return True
        return len(set(pred_str.split()).intersection(set(gt_str.split()))) > 0

    def evaluate_query(self, query_id, task_type, predictions):
        """Danh gia 1 cau query, tra ve final_score va mang R@k."""
        if query_id not in self.gt.get(task_type, {}):
            return 0.0, [0.0] * 5
            
        gt_item = self.gt[task_type][query_id]
        gt_video = gt_item["video_id"]
        
        r_scores = []
        for pred in predictions:
            if pred.get("video_id") != gt_video:
                r_scores.append(0.0)
                continue
                
            if task_type == "task1":
                try:
                    pred_frame = int(pred.get("frame_id", -1))
                    if int(gt_item["frame_start"]) <= pred_frame <= int(gt_item["frame_end"]):
                        r_scores.append(1.0)
                    else:
                        r_scores.append(0.0)
                except ValueError:
                    r_scores.append(0.0)
                    
            elif task_type == "task2":
                try:
                    pred_frame = int(pred.get("frame_id", -1))
                    if (int(gt_item["frame_start"]) <= pred_frame <= int(gt_item["frame_end"])) and self.check_answer_match(pred.get("answer", ""), gt_item["answer"]):
                        r_scores.append(1.0)
                    else:
                        r_scores.append(0.0)
                except ValueError:
                    r_scores.append(0.0)
                    
            elif task_type == "task3":
                events = gt_item["events"]
                pred_frames = pred.get("frame_ids", [])
                if len(pred_frames) != len(events):
                    r_scores.append(0.0)
                    continue
                    
                match_count = 0
                for idx, event in enumerate(events):
                    try:
                        if int(event["frame_start"]) <= int(pred_frames[idx]) <= int(event["frame_end"]):
                            match_count += 1
                    except ValueError:
                        pass
                r_scores.append(match_count / len(events))
                
        while len(r_scores) < 100:
            r_scores.append(0.0)
            
        k_values = [1, 5, 20, 50, 100]
        r_at_k = [float(max(r_scores[:k])) for k in k_values]
        return float(np.mean(r_at_k)), r_at_k

    def evaluate_all(self, predictions_dict):
        """Tinh diem trung binh toan bo tat ca cac Task."""
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
        return {
            "Final_Score": float(np.mean(all_final_scores)),
            "R@1": float(r_at_k_avg[0]),
            "R@5": float(r_at_k_avg[1]),
            "R@20": float(r_at_k_avg[2]),
            "R@50": float(r_at_k_avg[3]),
            "R@100": float(r_at_k_avg[4])
        }
