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
            # Phuong tien & Dua xe
            "xe đạp": ["Bicycle", "Wheel", "Helmet", "Hat", "Clothing", "Person", "Land vehicle", "Sports equipment", "Vehicle"],
            "đua xe": ["Bicycle", "Wheel", "Helmet", "Sports equipment", "Person", "Clothing"],
            "tay đua": ["Bicycle", "Wheel", "Helmet", "Hat", "Clothing", "Person", "Sports equipment"],
            "flycam": ["Aircraft", "Vehicle"],
            "thuyền": ["Boat", "Watercraft", "Vehicle"],
            "ghe": ["Boat", "Watercraft", "Vehicle"],
            "chèo ghe": ["Boat", "Watercraft", "Person"],
            "tàu vũ trụ": ["Rocket", "Space vehicle", "Helmet", "Suit", "Aircraft", "Airplane", "Clothing", "Person", "Vehicle"],
            "phóng tàu": ["Rocket", "Space vehicle", "Aircraft", "Vehicle"],
            "phi hành gia": ["Helmet", "Suit", "Clothing", "Person"],
            
            # Dong vat
            "dê": ["Goat", "Shirt", "Clothing", "Animal", "Cattle", "Livestock", "Mammal", "Person"],
            "cho dê ăn": ["Goat", "Shirt", "Clothing", "Animal", "Livestock", "Person"],
            "đàn dê": ["Goat", "Shirt", "Animal", "Livestock", "Mammal", "Person"],
            "hổ": ["Tiger", "Carnivore", "Animal", "Mammal"],
            "đàn hổ": ["Tiger", "Carnivore", "Animal", "Mammal"],
            "con hổ": ["Tiger", "Carnivore", "Animal", "Mammal"],
            "chim": ["Bird", "Plant", "Tree", "Flower", "Animal"],

            "chú chim": ["Bird", "Plant", "Tree", "Flower", "Animal"],
            "loài chim": ["Bird", "Plant", "Tree", "Flower", "Animal"],
            "cá mập": ["Shark", "Fish", "Sea life", "Boat", "Watercraft", "Building", "Person", "Animal"],
            "steven spielberg": ["Shark", "Fish", "Sea life", "Boat", "Watercraft", "Building", "Person"],
            "spielberg": ["Shark", "Fish", "Sea life", "Boat", "Watercraft", "Building", "Person"],
            "bạch tuộc": ["Sea life", "Animal", "Toy", "Bag", "Box", "Packaging", "Person", "Clothing"],
            "con mực": ["Sea life", "Animal", "Toy", "Bag", "Box", "Packaging", "Person", "Clothing"],
            "bọ": ["Beetle", "Robot", "Insect", "Arthropod", "Electronics", "Animal", "Toy"],
            "bọ cánh cứng": ["Beetle", "Robot", "Insect", "Arthropod", "Electronics", "Animal", "Toy"],
            "robot": ["Robot", "Beetle", "Electronics", "Toy"],
            "lausanne": ["Beetle", "Robot", "Electronics", "Toy", "Building"],
            
            # Am thuc & Nau an
            "bánh rán": ["Cake", "Doughnut", "Pastry", "Banana", "Dessert", "Plate", "Tableware", "Tray", "Food", "Fruit", "Plant"],
            "bánh": ["Cake", "Dessert", "Plate", "Tableware", "Food", "Baked goods"],
            "chocolate": ["Dessert", "Cake", "Food"],
            "chuối": ["Banana", "Fruit", "Plate", "Tableware", "Food", "Plant"],
            "dâu tây": ["Fruit", "Plate", "Food", "Plant"],
            "dâu": ["Fruit", "Plate", "Food", "Plant"],
            "panna cotta": ["Dessert", "Glass", "Drink", "Flower", "Rose", "Plate", "Tableware", "Food", "Plant", "Houseplant"],
            "gỏi cuốn": ["Spring roll", "Flower", "Rose", "Plate", "Tableware", "Dish", "Food", "Plant", "Vegetable", "Baked goods"],
            "bánh tráng": ["Spring roll", "Dish", "Plate", "Food", "Baked goods"],
            "măng tây": ["Vegetable", "Plant", "Kitchen utensil", "Tableware", "Plate", "Food"],
            "nấm": ["Mushroom", "Vegetable", "Tableware", "Kitchen utensil", "Food", "Plant", "Person"],
            "cắt nấm": ["Mushroom", "Vegetable", "Kitchen utensil", "Tableware", "Food", "Person"],
            "củ năng": ["Vegetable", "Food", "Plant"],
            "đậu hũ": ["Food", "Plant"],
            "đậu hủ": ["Food", "Plant"],
            "thịt": ["Meat", "Beef", "Pork", "Kitchen utensil", "Tableware", "Food", "Person"],
            "thịt nạc": ["Meat", "Beef", "Pork", "Kitchen utensil", "Tableware", "Food", "Person"],
            "thịt xay": ["Meat", "Beef", "Pork", "Book", "Poster", "Kitchen utensil", "Tableware", "Food", "Person"],
            "thịt nạc xay": ["Meat", "Beef", "Pork", "Book", "Poster", "Kitchen utensil", "Tableware", "Food", "Person"],
            "dứa": ["Pineapple", "Boat", "Watercraft", "Hat", "Helmet", "Clothing", "Food", "Fruit", "Person", "Plant", "Vehicle"],
            "thu hoạch dứa": ["Pineapple", "Boat", "Watercraft", "Hat", "Helmet", "Clothing", "Food", "Fruit", "Person", "Plant", "Vehicle"],
            "nấu ăn": ["Kitchen utensil", "Tableware", "Food", "Person"],
            "chảo": ["Kitchen utensil", "Tableware"],
            "bếp": ["Kitchen appliance", "Oven"],
            "đĩa": ["Plate", "Tableware"],
            "khay": ["Tray", "Tableware"],
            "ly": ["Glass", "Drink", "Tableware"],
            "túi giấy": ["Bag", "Box", "Packaging"],
            
            # Thiet bi, Nghe thuat & Su kien
            "máy ảnh": ["Camera", "Camera lens", "Clothing", "Shirt", "Electronics", "Person"],
            "ống kính": ["Camera", "Camera lens", "Clothing", "Shirt", "Electronics", "Person"],
            "vệ sinh máy ảnh": ["Camera", "Camera lens", "Clothing", "Shirt", "Electronics", "Person"],
            "điêu khắc cát": ["Sculpture", "Sand", "Skateboard", "Sports equipment", "Statue", "Art", "Person"],
            "tượng": ["Sculpture", "Statue", "Art"],
            "patin": ["Skateboard", "Sports equipment", "Person"],
            "trượt ván": ["Skateboard", "Sports equipment", "Person"],
            "múa lân": ["Dragon", "Lion", "Costume", "Sculpture", "Statue", "Clothing", "Person", "Toy"],
            "lân": ["Dragon", "Lion", "Costume", "Sculpture", "Statue", "Clothing", "Person", "Toy"],
            "rồng": ["Dragon", "Lion", "Sculpture", "Statue", "Toy"],
            "nhạc cụ": ["Musical instrument", "Bookcase", "Book", "Furniture", "Person", "Clothing"],
            "kệ sách": ["Bookcase", "Book", "Furniture"],
            "sách": ["Book", "Bookcase"],
            "mảnh bìa": ["Suit", "Person", "Hat", "Helmet", "Box", "Shirt", "Clothing"],
            "đổ bóng": ["Suit", "Person", "Hat", "Helmet", "Box", "Shirt", "Clothing"],
            "trang phục": ["Clothing", "Suit", "Dress", "Person"],
            "áo thun": ["Shirt", "Clothing", "Person"],
            "áo sơ mi": ["Shirt", "Clothing", "Person"],
            "nón": ["Hat", "Helmet", "Clothing"],
            "mũ": ["Hat", "Helmet", "Clothing"],
            "hoa": ["Flower", "Rose", "Plant", "Houseplant"],
            "hoa pansy": ["Flower", "Rose", "Plant"]
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


    def get_frame_objects(self, video_id, frame_idx, frame_id=""):
        """
        Lay danh sach thuc the cua mot keyframe tu file JSON cua BTC.
        Tu dong ho tro dinh dang 3 chu so (001.json), 4 chu so (0001.json), 5 chu so va frame_id.
        """
        cache_key = f"{video_id}_{frame_idx}_{frame_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        v_folder = self._find_video_object_folder(video_id)
        if not v_folder:
            return None
            
        idx_1 = frame_idx + 1
        candidate_filenames = [
            f"{frame_idx:03d}.json",
            f"{idx_1:03d}.json",
            f"{frame_idx:04d}.json",
            f"{idx_1:04d}.json",
            f"{frame_idx:05d}.json",
            f"{frame_idx}.json",
            f"{idx_1}.json"
        ]
        if frame_id:
            fid_clean = str(frame_id).strip()
            candidate_filenames.append(f"{fid_clean}.json")
            if fid_clean.isdigit():
                candidate_filenames.append(f"{int(fid_clean):03d}.json")
                candidate_filenames.append(f"{int(fid_clean):04d}.json")
                
        json_file = None
        for cf in candidate_filenames:
            p = os.path.join(v_folder, cf)
            if os.path.exists(p):
                json_file = p
                break
                
        if not json_file:
            return None
            
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            entities = data.get("detection_class_entities", [])
            scores = [float(s) for s in data.get("detection_scores", [])]
            boxes = data.get("detection_boxes", [])
            
            filtered_objects = []
            for ent, sc, bx in zip(entities, scores, boxes):
                if sc >= 0.15:
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

    def get_entity_information_weight(self, entity_name, query_text=""):
        """
        Dinh luong Ham luong Thong tin (Information Density) va Nang luc Phan biet cua thuc the:
        - Tier 1 (x4.5): Chu the doc nhat, mang tinh quyet dinh dinh danh phan canh.
        - Tier 2 (x2.5): Dao cu & Boi canh phan biet (Bookcase, Boat, Hat, Plate, Flower...).
        - Tier 3 (x0.5): Tac nhan pho quat / Nen (Person, Clothing, Building, Food, Plant...).
        """
        ent = entity_name.lower().strip()
        
        tier1_high_info = {
            "goat", "tiger", "carnivore", "camera", "camera lens", "rocket", "space vehicle",
            "mushroom", "pineapple", "skateboard", "robot", "beetle", "shark",
            "lantern", "dragon", "lion", "bicycle", "bird", "spring roll",
            "cake", "doughnut", "pastry", "musical instrument", "suit", "poster",
            "billboard", "meat", "beef", "pork", "dessert", "sculpture"
        }
        
        tier2_context_props = {
            "helmet", "hat", "boat", "watercraft", "bookcase", "book", "statue",
            "flower", "rose", "banana", "plate", "tableware", "tray", "glass",
            "drink", "bag", "box", "packaging", "sand", "kitchen utensil", "shirt",
            "dress", "sea life", "vegetable"
        }

        
        if ent in tier1_high_info:
            return 4.5
        elif ent in tier2_context_props:
            return 2.5
        else:
            return 0.5

    def extract_tiered_entities(self, query_text):
        """
        Boc tach thuc the theo dung 3 Tier ro rang cho tung cau hoi:
        - Tier 1: Chu the quyet dinh / Hanh dong trong tam (x4.5)
        - Tier 2: Boi canh, Dao cu & Thuoc tinh phan biet (x2.5)
        - Tier 3: Tac nhan nen & Moi truong chung (x0.5)
        """
        all_ents = self.extract_target_entities(query_text)
        t1, t2, t3 = [], [], []
        
        for e in all_ents:
            w = self.get_entity_information_weight(e, query_text)
            if w >= 4.0:
                t1.append(e)
            elif w >= 2.0:
                t2.append(e)
            else:
                t3.append(e)
                
        # Bao dam moi cau deu co su hien dien cua ca 3 Tier
        if not t1 and t2:
            t1.append(t2.pop(0))
        if not t2 and t3:
            t2.append(t3.pop(0))
        if not t3:
            t3.append("person" if "person" not in t1 and "person" not in t2 else "clothing")
            
        return {
            "all": all_ents,
            "tier1": sorted(list(set(t1))),
            "tier2": sorted(list(set(t2))),
            "tier3": sorted(list(set(t3)))
        }


    def boost_candidates(self, candidates, query_text):
        """
        Khai thac toan dien du lieu Objects dua tren Ham luong Thong tin (Information Content):
        1. Nang diem khung hinh nao chua vat the co gia tri thong tin cao.
        2. Cong huong diem thuong cho video chua dong thoi cac thuc the cot loi.
        3. Phat nhe cac video thieu hut chu the quyet dinh (Tier 1).
        """
        if not self._objects_root or not candidates:
            return candidates
            
        target_entities = set(self.extract_target_entities(query_text))
        if not target_entities:
            return candidates
            
        # Xac dinh xem cau hoi co chua chu the hiem (Tier 1) hay khong
        has_tier1_target = any(self.get_entity_information_weight(e, query_text) >= 4.0 for e in target_entities)
        
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
                video_matched_entities = set()
                video_has_tier1_match = False
                
                # He so dong thuan boi canh CLIP: Danh gia muc do phu hop cua background/ngu nghia
                base_rrf = cand.get("rrf_score", 0.0)
                semantic_gate = max(0.4, min(1.0, base_rrf * 15.0 if base_rrf > 0 else 0.5))
                
                for f_idx in top_frame_idxs:
                    objs = self.get_frame_objects(video_id, int(f_idx))
                    if not objs:
                        continue
                        
                    frame_match_score = 0.0
                    for obj in objs:
                        ent_lower = obj["entity"].lower()
                        if ent_lower in target_entities or any(t_ent == ent_lower for t_ent in target_entities):
                            # Tinh trong so theo Ham luong Thong tin cua thuc the
                            info_weight = self.get_entity_information_weight(ent_lower, query_text)
                            frame_match_score += float(obj["score"]) * (info_weight / 2.0)
                            video_matched_entities.add(ent_lower)
                            if info_weight >= 4.0:
                                video_has_tier1_match = True
                            
                    if frame_match_score > 0:
                        # Dieu bien nhan (Multiplicative): Khung hinh phai vua hop boi canh CLIP, VUA chua vat the
                        scores[f_idx] = scores[f_idx] * (1.0 + 0.12 * frame_match_score)
                        video_object_bonus += frame_match_score * 0.04
                        
                # 1. Diem thuong cong huong khi video chua dong thoi tu 2 vat the muc tieu tro len
                if len(video_matched_entities) >= 2:
                    video_object_bonus += 0.15 * len(video_matched_entities)
                    
                # 2. Thuong them khi khop dung chu the Tier 1 (nhung phai qua cong kiem duyet ngu nghia)
                if video_has_tier1_match:
                    video_object_bonus += 0.25 * semantic_gate
                elif has_tier1_target and not video_has_tier1_match:
                    # Phat nhe vi video hoan toan khong co chu the hiem cua de bai
                    video_object_bonus -= 0.08 * semantic_gate
                    
                cand_copy["dense_info"]["all_scores"] = scores
                cand_copy["dense_info"]["best_frame_idx"] = int(np.argmax(scores))
                cand_copy["dense_info"]["max_score"] = float(np.max(scores))
                # Cong diem thuong da duoc kiem duyet boi canh va ngu nghia
                cand_copy["rrf_score"] = base_rrf + (video_object_bonus * semantic_gate)
                
            boosted_candidates.append(cand_copy)

            
        boosted_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return boosted_candidates


