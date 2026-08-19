import os
import json
import re

class ObjectSearcher:
    """
    Module khai thac du lieu Object Detection JSON cua BTC sieu toc (< 0.01s).
    - Chi doc truc tiep cac frame dinh cua Top video ung vien (khong quet de quy hang trieu file).
    - Ho tro dem so luong vat the chinh xac cho Task 2 VQA.
    """
    def __init__(self, config):
        self.config = config
        self.objects_dir = config.get("data", {}).get("objects_dir", "")
        self._cache = {} # Cache doc file json cua cac video
        
    def _find_video_object_folder(self, video_id):
        """Tim truc tiep thu muc chua cac file JSON object cua video trong 0.0001s."""
        if not self.objects_dir or not os.path.exists(self.objects_dir):
            return None
            
        level = video_id.split('_')[0] if '_' in video_id else ""
        candidate_dirs = [
            os.path.join(self.objects_dir, video_id),
            os.path.join(self.objects_dir, "objects", video_id),
            os.path.join(self.objects_dir, f"objects_{level}", "objects", video_id),
            os.path.join(self.objects_dir, f"objects_{level}", video_id),
            os.path.join(self.objects_dir, level, video_id),
            os.path.join(os.path.dirname(self.objects_dir), "objects-aic25-b1", "objects", video_id),
            os.path.join(os.path.dirname(self.objects_dir), "objects", video_id)
        ]
        
        for c in candidate_dirs:
            if os.path.exists(c):
                return c
        return None

    def get_frame_objects(self, video_id, frame_idx):
        """
        Lay danh sach thuc the va bounding boxes cua mot keyframe cu the.
        """
        cache_key = f"{video_id}_{frame_idx}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        v_folder = self._find_video_object_folder(video_id)
        if not v_folder:
            return None
            
        # Tên file json thuong la 0000.json, 0001.json...
        json_file = os.path.join(v_folder, f"{frame_idx:04d}.json")
        if not os.path.exists(json_file):
            json_file = os.path.join(v_folder, f"{frame_idx}.json")
            
        if not os.path.exists(json_file):
            return None
            
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            entities = data.get("detection_class_entities", [])
            scores = [float(s) for s in data.get("detection_scores", [])]
            boxes = data.get("detection_boxes", [])
            
            filtered_objects = []
            for ent, sc, bx in zip(entities, scores, boxes):
                if sc >= 0.25:
                    filtered_objects.append({
                        "entity": ent,
                        "score": sc,
                        "box": bx
                    })
                    
            self._cache[cache_key] = filtered_objects
            return filtered_objects
        except Exception:
            return None

    def boost_candidates(self, candidates, query_en):
        """
        Tang diem thuong (Bonus Score) sieu toc (< 0.01s):
        Chi kiem tra 3 frame dinh cua Top 10 video dau bang.
        """
        if not self.objects_dir or not os.path.exists(self.objects_dir):
            return candidates
            
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_en.lower()))
        if not query_words:
            return candidates
            
        # Chi boost tren Top 10 video ung vien tiem nang nhat de giu toc do < 10ms
        for cand in candidates[:10]:
            video_id = cand["video_id"]
            dense_info = cand.get("dense_info")
            if not dense_info or "all_scores" not in dense_info:
                continue
                
            scores = dense_info["all_scores"]
            if len(scores) == 0:
                continue
                
            # Lay 3 chi so frame cao diem nhat
            import numpy as np
            top_frame_idxs = np.argsort(scores)[::-1][:3]
            
            bonus_total = 0.0
            for f_idx in top_frame_idxs:
                objs = self.get_frame_objects(video_id, int(f_idx))
                if not objs:
                    continue
                    
                for obj in objs:
                    ent_lower = obj["entity"].lower()
                    if any(w in ent_lower for w in query_words):
                        bonus_total += 0.05
                        break
                        
            if bonus_total > 0:
                cand["rrf_score"] = cand.get("rrf_score", 0.0) + min(0.15, bonus_total)
                
        # Sap xep lai sau khi boost
        candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return candidates

    def count_entities_for_vqa(self, video_id, frame_idx, target_class_keyword):
        """
        Dem so luong vat the thuoc mot lop cu the trong frame de ho tro cau hoi Task 2.
        """
        objs = self.get_frame_objects(video_id, frame_idx)
        if not objs:
            return None
            
        target_kw = target_class_keyword.lower()
        count = 0
        for obj in objs:
            if target_kw in obj["entity"].lower() and obj["score"] >= 0.3:
                count += 1
        return count
