import os
import glob
import json
import re
import numpy as np

class ObjectSearcher:
    """
    Module khai thac du lieu Object Detection JSON cua BTC de:
    1. Tim dung video chua dung cac vat the trong cau hoi (Video Boosting).
    2. Dinh vi chinh xac khung hinh vat ly chua vat the do (Frame-Level Object Grounding).
    """
    def __init__(self, config):
        self.config = config
        self.objects_dir = config.get("data", {}).get("objects_dir", "")
        self._objects_root = self._find_objects_root()
        self._cache = {}
        
        # Bang anh xa tu khoa Tieng Viet / Tieng Anh sang cac Class Entities cua BTC
        self.entity_map = {
            "xe đạp": ["Bicycle", "Land vehicle", "Vehicle", "Wheel", "Person"],
            "đua xe": ["Bicycle", "Person", "Sports equipment", "Helmet"],
            "dê": ["Goat", "Animal", "Livestock", "Cattle", "Mammal"],
            "cho dê ăn": ["Goat", "Animal", "Person", "Livestock"],
            "bánh rán": ["Cake", "Food", "Baked goods", "Dessert", "Pastry", "Doughnut"],
            "bánh": ["Cake", "Food", "Baked goods", "Dessert"],
            "hoa": ["Flower", "Plant", "Rose", "Houseplant"],
            "pansy": ["Flower", "Plant"],
            "máy ảnh": ["Camera", "Electronics", "Camera lens"],
            "ống kính": ["Camera", "Camera lens", "Electronics"],
            "vệ sinh máy ảnh": ["Camera", "Camera lens", "Person"],
            "thuyền": ["Boat", "Watercraft", "Vehicle"],
            "ghe": ["Boat", "Watercraft", "Vehicle"],
            "tàu vũ trụ": ["Airplane", "Rocket", "Aircraft", "Space vehicle", "Vehicle"],
            "phi hành gia": ["Person", "Clothing", "Suit"],
            "hổ": ["Tiger", "Cat", "Carnivore", "Animal", "Mammal"],
            "đàn hổ": ["Tiger", "Cat", "Animal", "Mammal"],
            "nấm": ["Mushroom", "Food", "Plant", "Vegetable"],
            "cắt nấm": ["Mushroom", "Food", "Kitchen utensil", "Person"],
            "panna cotta": ["Dessert", "Food", "Drink", "Tableware", "Glass"],
            "măng tây": ["Vegetable", "Food", "Plant"],
            "điêu khắc cát": ["Sculpture", "Sand", "Art", "Statue", "Person"],
            "múa lân": ["Person", "Clothing", "Costume", "Dragon", "Lion"],
            "rồng": ["Dragon", "Sculpture", "Statue", "Toy"],
            "thịt": ["Meat", "Food", "Beef", "Pork"],
            "gỏi cuốn": ["Food", "Vegetable", "Spring roll", "Dish"],
            "dứa": ["Pineapple", "Fruit", "Food", "Plant"],
            "thu hoạch dứa": ["Fruit", "Food", "Plant", "Person", "Boat"],
            "cá mập": ["Shark", "Fish", "Animal", "Sea life"],
            "bọ": ["Insect", "Arthropod", "Beetle", "Animal"],
            "robot": ["Robot", "Toy", "Electronics"]
        }
        
    def _find_objects_root(self):
        """Tu dong quet va xac dinh thu muc objects trong he thong."""
        candidate_roots = [
            self.objects_dir,
            os.path.join(os.path.dirname(self.objects_dir), "objects") if self.objects_dir else None,
            "/kaggle/input/ai-challenge-hcmc-2026-objects/objects",
            "/kaggle/input/ai-challenge-hcmc-2026-objects",
            "/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-objects/objects",
            "/kaggle/input/ai-challenge-hcmc-2026-metadata/objects-aic25-b1/objects",
            "/kaggle/input/ai-challenge-hcmc-2026-metadata/objects"
        ]
        
        for r in candidate_roots:
            if r and os.path.exists(r):
                print(f"ObjectSearcher: Tim thay thu muc Objects tai: {r}")
                return r
                
        if os.path.exists("/kaggle/input"):
            for root, dirs, _ in os.walk("/kaggle/input"):
                if "objects" in root.lower() and len(dirs) > 5:
                    print(f"ObjectSearcher: Tu dong phat hien thu muc Objects tai: {root}")
                    return root
                    
        print("ObjectSearcher: Canh bao: Chua tim thay thu muc Objects tren he thong.")
        return None

    def _find_video_object_folder(self, video_id):
        """Tim thu muc chua cac file JSON object cua video."""
        if not self._objects_root:
            return None
            
        level = video_id.split('_')[0] if '_' in video_id else ""
        candidate_dirs = [
            os.path.join(self._objects_root, video_id),
            os.path.join(self._objects_root, "objects", video_id),
            os.path.join(self._objects_root, f"objects_{level}", "objects", video_id),
            os.path.join(self._objects_root, f"objects_{level}", video_id),
            os.path.join(self._objects_root, level, video_id),
            os.path.join(self._objects_root, "objects-aic25-b1", "objects", video_id)
        ]
        
        for c in candidate_dirs:
            if os.path.exists(c):
                return c
                
        # Quet fallback
        matches = glob.glob(os.path.join(self._objects_root, f"**/{video_id}"), recursive=True)
        if matches and os.path.isdir(matches[0]):
            return matches[0]
            
        return None

    def get_frame_objects(self, video_id, frame_idx):
        """Lay danh sach thuc the cua mot keyframe."""
        cache_key = f"{video_id}_{frame_idx}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        v_folder = self._find_video_object_folder(video_id)
        if not v_folder:
            return None
            
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
                if sc >= 0.20:
                    filtered_objects.append({
                        "entity": ent,
                        "score": sc,
                        "box": bx
                    })
                    
            self._cache[cache_key] = filtered_objects
            return filtered_objects
        except Exception:
            return None

    def extract_target_entities(self, query_text):
        """Trich xuat danh sach entity can tim tu cau hoi."""
        text_lower = query_text.lower()
        target_entities = set()
        
        for kw, ents in self.entity_map.items():
            if kw in text_lower:
                for e in ents:
                    target_entities.add(e.lower())
                    
        # Trich xuat them cac tu don
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        for w in words:
            target_entities.add(w.lower())
            
        return list(target_entities)

    def boost_candidates(self, candidates, query_text):
        """
        Khai thac toan dien du lieu Objects:
        1. Nang diem khung hinh nao chua dung vat the muc tieu (Frame Grounding).
        2. Cong diem thuong cho video chua dung vat the do (Video Re-ranking).
        """
        if not self._objects_root or not candidates:
            return candidates
            
        target_entities = self.extract_target_entities(query_text)
        if not target_entities:
            return candidates
            
        boosted_candidates = []
        
        for rank, cand in enumerate(candidates):
            cand_copy = dict(cand)
            video_id = cand["video_id"]
            dense_info = cand.get("dense_info")
            
            if not dense_info or "all_scores" not in dense_info:
                boosted_candidates.append(cand_copy)
                continue
                
            v_folder = self._find_video_object_folder(video_id)
            if not v_folder:
                boosted_candidates.append(cand_copy)
                continue
                
            # Copy all_scores de cap nhat Frame-Level Boost
            scores = np.array(dense_info["all_scores"], dtype=np.float32)
            n_frames = len(scores)
            
            # Chi kiem tra cac frame co diem cao trong Top 25 video dau bang
            if rank < 30:
                top_frame_idxs = np.argsort(scores)[::-1][:12]
                video_object_bonus = 0.0
                
                for f_idx in top_frame_idxs:
                    objs = self.get_frame_objects(video_id, int(f_idx))
                    if not objs:
                        continue
                        
                    frame_match_score = 0.0
                    for obj in objs:
                        ent_lower = obj["entity"].lower()
                        if any(t_ent in ent_lower or ent_lower in t_ent for t_ent in target_entities):
                            frame_match_score += float(obj["score"])
                            
                    if frame_match_score > 0:
                        # Cong truc tiep vao frame score tai dung vi tri vat the xuat hien
                        scores[f_idx] += frame_match_score * 0.20
                        video_object_bonus += frame_match_score * 0.05
                        
                # Cap nhat lai dense_info
                cand_copy["dense_info"]["all_scores"] = scores
                cand_copy["dense_info"]["best_frame_idx"] = int(np.argmax(scores))
                cand_copy["dense_info"]["max_score"] = float(np.max(scores))
                
                # Cong diem thuong RRF
                cand_copy["rrf_score"] = cand.get("rrf_score", 0.0) + video_object_bonus
                
            boosted_candidates.append(cand_copy)
            
        # Sap xep lai danh sach theo rrf_score da duoc boost
        boosted_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return boosted_candidates
