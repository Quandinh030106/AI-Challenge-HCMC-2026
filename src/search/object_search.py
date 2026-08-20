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
        
        # Bang anh xa thuc the toan dien cho tat ca 24 cau de thi sang cac lop OpenImages chuan cua BTC
        self.entity_map = {
            # Phuong tien & Giao thong
            "xe đạp": ["Bicycle", "Land vehicle", "Vehicle", "Wheel", "Person"],
            "đua xe": ["Bicycle", "Person", "Sports equipment", "Helmet"],
            "tay đua": ["Person", "Bicycle", "Helmet", "Sports equipment", "Clothing"],
            "flycam": ["Aircraft", "Vehicle"],
            "thuyền": ["Boat", "Watercraft", "Vehicle"],
            "ghe": ["Boat", "Watercraft", "Vehicle"],
            "chèo ghe": ["Boat", "Watercraft", "Person"],
            "tàu vũ trụ": ["Airplane", "Rocket", "Aircraft", "Space vehicle", "Vehicle"],
            "phóng tàu": ["Rocket", "Aircraft", "Space vehicle"],
            "phi hành gia": ["Person", "Clothing", "Suit", "Helmet"],
            
            # Dong vat
            "dê": ["Goat", "Animal", "Livestock", "Cattle", "Mammal"],
            "cho dê ăn": ["Goat", "Animal", "Person", "Livestock"],
            "đàn dê": ["Goat", "Animal", "Livestock", "Mammal"],
            "hổ": ["Tiger", "Cat", "Carnivore", "Animal", "Mammal"],
            "đàn hổ": ["Tiger", "Cat", "Carnivore", "Animal", "Mammal"],
            "con hổ": ["Tiger", "Cat", "Carnivore", "Animal", "Mammal"],
            "chim": ["Bird", "Animal"],
            "chú chim": ["Bird", "Animal"],
            "loài chim": ["Bird", "Animal"],
            "cá mập": ["Shark", "Fish", "Animal", "Sea life"],
            "bạch tuộc": ["Animal", "Food", "Sea life"],
            "con mực": ["Animal", "Food", "Sea life"],
            "bọ": ["Insect", "Arthropod", "Beetle", "Animal"],
            "bọ cánh cứng": ["Insect", "Arthropod", "Beetle", "Animal"],
            "robot": ["Robot", "Toy", "Electronics"],
            
            # Am thuc & Nau an
            "bánh rán": ["Cake", "Food", "Baked goods", "Dessert", "Pastry", "Doughnut"],
            "bánh": ["Cake", "Food", "Baked goods", "Dessert"],
            "chocolate": ["Food", "Dessert"],
            "chuối": ["Banana", "Fruit", "Food", "Plant"],
            "dâu tây": ["Fruit", "Food", "Plant"],
            "dâu": ["Fruit", "Food", "Plant"],
            "panna cotta": ["Dessert", "Food", "Drink", "Tableware", "Glass"],
            "gỏi cuốn": ["Food", "Vegetable", "Spring roll", "Dish"],
            "bánh tráng": ["Food", "Dish", "Baked goods"],
            "măng tây": ["Vegetable", "Food", "Plant"],
            "nấm": ["Mushroom", "Food", "Plant", "Vegetable"],
            "cắt nấm": ["Mushroom", "Food", "Kitchen utensil", "Person"],
            "củ năng": ["Vegetable", "Food", "Plant"],
            "đậu hũ": ["Food", "Plant"],
            "đậu hủ": ["Food", "Plant"],
            "thịt": ["Meat", "Food", "Beef", "Pork"],
            "thịt nạc": ["Meat", "Food", "Beef", "Pork"],
            "thịt xay": ["Meat", "Food", "Beef", "Pork"],
            "dứa": ["Pineapple", "Fruit", "Food", "Plant"],
            "thu hoạch dứa": ["Fruit", "Food", "Plant", "Person", "Boat"],
            "nấu ăn": ["Food", "Kitchen utensil", "Person", "Tableware"],
            "chảo": ["Kitchen utensil", "Tableware"],
            "bếp": ["Kitchen appliance", "Oven"],
            "đĩa": ["Tableware", "Plate"],
            "khay": ["Tableware", "Tray"],
            "ly": ["Drink", "Glass", "Tableware"],
            "túi giấy": ["Bag", "Box", "Packaging"],
            
            # Thiet bi & Nghe thuat
            "máy ảnh": ["Camera", "Electronics", "Camera lens"],
            "ống kính": ["Camera", "Camera lens", "Electronics"],
            "vệ sinh máy ảnh": ["Camera", "Camera lens", "Person"],
            "điêu khắc cát": ["Sculpture", "Sand", "Art", "Statue", "Person"],
            "tượng": ["Sculpture", "Statue", "Art"],
            "patin": ["Sports equipment", "Person"],
            "trượt ván": ["Skateboard", "Sports equipment", "Person"],
            "múa lân": ["Person", "Clothing", "Costume", "Dragon", "Lion"],
            "lân": ["Person", "Costume", "Dragon", "Lion"],
            "rồng": ["Dragon", "Sculpture", "Statue", "Toy"],
            "nhạc cụ": ["Musical instrument", "Person"],
            "kệ sách": ["Bookcase", "Furniture", "Book"],
            "sách": ["Book", "Bookcase"],
            "trang phục": ["Clothing", "Suit", "Dress", "Person"],
            "áo thun": ["Clothing", "Shirt", "Person"],
            "áo sơ mi": ["Clothing", "Shirt", "Person"],
            "nón": ["Hat", "Helmet", "Clothing"],
            "mũ": ["Hat", "Helmet", "Clothing"],
            "hoa": ["Flower", "Plant", "Rose", "Houseplant"],
            "hoa pansy": ["Flower", "Plant"]
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
            os.path.join(self._objects_root, "objects-aic25-b1", "objects", video_id),
            os.path.join(self._objects_root, "objects-aic25-b1", video_id)
        ]
        
        for c in candidate_dirs:
            if os.path.exists(c):
                return c
                
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
        """
        Trich xuat danh sach nhan vat the OpenImages hop le tu cau hoi.
        Chi giu lai cac lop vat the thi giac thuc su cua BTC, loai bo 100% am tiet rac tieng Viet.
        """
        text_lower = query_text.lower()
        target_entities = set()
        
        # 1. Khop tu khoa Tieng Viet qua Bang Anh Xa Thuc The
        for kw, ents in self.entity_map.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                for e in ents:
                    target_entities.add(e.lower())
                    
        # 2. Khop truc tiep cac lop OpenImages pho bien neu xuat hien trong cau dich Tieng Anh
        valid_openimages_classes = {
            "person", "man", "woman", "girl", "boy", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "watercraft", "traffic light", "fire hydrant", "stop sign", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "goat",
            "tiger", "lion", "carnivore", "mammal", "animal", "shark", "fish", "sea life", "insect", "beetle",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "food", "baked goods", "dessert", "pastry",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "camera", "camera lens", "electronics",
            "rocket", "aircraft", "space vehicle", "vehicle", "clothing", "suit", "helmet", "sculpture", "statue",
            "art", "sand", "flower", "plant", "rose", "mushroom", "vegetable", "meat", "beef", "pork", "fruit",
            "pineapple", "dish", "spring roll", "robot", "toy", "glass", "tableware", "kitchen utensil"
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        for w in words:
            if w in valid_openimages_classes:
                target_entities.add(w)
                
        return sorted(list(target_entities))

    def boost_candidates(self, candidates, query_text):
        """
        Khai thac toan dien du lieu Objects:
        1. Nang diem khung hinh nao chua dung vat the muc tieu (Frame Grounding).
        2. Cong diem thuong cho video chua dung vat the do (Video Re-ranking).
        """
        if not self._objects_root or not candidates:
            return candidates
            
        target_entities = set(self.extract_target_entities(query_text))
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
                
            scores = np.array(dense_info["all_scores"], dtype=np.float32)
            
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
                        # Khop chinh xac (Exact Matching) hoac khop ten lop chuan
                        if ent_lower in target_entities or any(t_ent == ent_lower for t_ent in target_entities):
                            frame_match_score += float(obj["score"])
                            
                    if frame_match_score > 0:
                        scores[f_idx] += frame_match_score * 0.20
                        video_object_bonus += frame_match_score * 0.05
                        
                cand_copy["dense_info"]["all_scores"] = scores
                cand_copy["dense_info"]["best_frame_idx"] = int(np.argmax(scores))
                cand_copy["dense_info"]["max_score"] = float(np.max(scores))
                cand_copy["rrf_score"] = cand.get("rrf_score", 0.0) + video_object_bonus
                
            boosted_candidates.append(cand_copy)
            
        boosted_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return boosted_candidates

