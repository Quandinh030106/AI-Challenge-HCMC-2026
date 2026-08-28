import os
import glob
import json
import re
import numpy as np
from src.utils import natural_sort_key

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
            # Tu dien danh tu thuc the don thuan (khong doan mo boi canh)
            "người": ["Person"],
            "nam giới": ["Man", "Person"],
            "nữ giới": ["Woman", "Person"],
            "phụ nữ": ["Woman", "Person"],
            "đàn ông": ["Man", "Person"],
            "trẻ em": ["Girl", "Boy", "Person"],
            "cô gái": ["Girl", "Woman", "Person"],
            "chàng trai": ["Boy", "Man", "Person"],
            "con trai": ["Boy", "Man", "Person"],
            "con gái": ["Girl", "Woman", "Person"],

            "xe đạp": ["Bicycle", "Vehicle"],
            "xe máy": ["Motorcycle", "Vehicle"],
            "ô tô": ["Car", "Vehicle"],
            "xe hơi": ["Car", "Vehicle"],
            "xe buýt": ["Bus", "Vehicle"],
            "xe tải": ["Truck", "Vehicle"],
            "máy bay": ["Airplane", "Aircraft", "Vehicle"],
            "trực thăng": ["Helicopter", "Aircraft", "Vehicle"],
            "tàu hỏa": ["Train", "Vehicle"],
            "thuyền": ["Boat", "Watercraft", "Vehicle"],
            "tàu thủy": ["Boat", "Watercraft", "Vehicle"],
            "tàu vũ trụ": ["Rocket", "Space vehicle", "Vehicle"],
            "tên lửa": ["Rocket", "Space vehicle"],
            "phi hành gia": ["Person", "Suit", "Helmet"],
            "chó": ["Dog", "Animal", "Mammal", "Carnivore"],
            "mèo": ["Cat", "Animal", "Mammal", "Carnivore"],
            "ngựa": ["Horse", "Animal", "Mammal"],
            "bò": ["Cow", "Cattle", "Livestock", "Animal", "Mammal"],
            "dê": ["Goat", "Livestock", "Animal", "Mammal"],
            "cừu": ["Sheep", "Livestock", "Animal", "Mammal"],
            "hổ": ["Tiger", "Carnivore", "Animal", "Mammal"],
            "sư tử": ["Lion", "Carnivore", "Animal", "Mammal"],
            "voi": ["Elephant", "Animal", "Mammal"],
            "gấu": ["Bear", "Carnivore", "Animal", "Mammal"],
            "hươu": ["Deer", "Animal", "Mammal"],
            "chim": ["Bird", "Animal"],
            "cá": ["Fish", "Sea life", "Animal"],
            "cá mập": ["Shark", "Fish", "Sea life", "Animal"],
            "bạch tuộc": ["Sea life", "Animal"],
            "mực": ["Sea life", "Animal"],
            "bọ cánh cứng": ["Beetle", "Insect", "Animal"],
            "côn trùng": ["Insect", "Animal"],
            "bánh": ["Cake", "Baked goods", "Food"],
            "bánh rán": ["Doughnut", "Cake", "Pastry", "Food"],
            "chuối": ["Banana", "Fruit", "Food"],
            "táo": ["Apple", "Fruit", "Food"],
            "dâu": ["Fruit", "Food"],
            "dứa": ["Pineapple", "Fruit", "Food"],
            "nấm": ["Mushroom", "Vegetable", "Food"],
            "măng tây": ["Vegetable", "Food"],
            "thịt": ["Meat", "Beef", "Pork", "Food"],
            "gỏi cuốn": ["Spring roll", "Dish", "Food"],
            "pizza": ["Pizza", "Food"],
            "máy ảnh": ["Camera", "Camera lens", "Electronics"],
            "ống kính": ["Camera lens", "Camera"],
            "điện thoại": ["Cell phone", "Electronics"],
            "máy tính": ["Laptop", "Computer", "Electronics"],
            "tivi": ["Television", "Electronics"],
            "đàn": ["Musical instrument"],
            "nhạc cụ": ["Musical instrument"],
            "trống": ["Drum", "Musical instrument"],
            "kèn": ["Musical instrument"],
            "trượt ván": ["Skateboard", "Sports equipment"],
            "ván trượt": ["Skateboard", "Sports equipment"],
            "bóng đá": ["Sports ball", "Sports equipment"],
            "múa lân": ["Lion", "Dragon", "Costume"],
            "tượng": ["Sculpture", "Statue", "Art"],
            "điêu khắc": ["Sculpture", "Statue", "Art"],
            "hoa": ["Flower", "Rose", "Plant"],
            "cây": ["Tree", "Plant"],
            "sách": ["Book"],
            "kệ sách": ["Bookcase", "Furniture"],
            "bàn": ["Table", "Furniture"],
            "ghế": ["Chair", "Furniture"],
            "áo": ["Shirt", "Clothing"],
            "quần": ["Pants", "Clothing"],
            "nón": ["Hat", "Helmet", "Clothing"],
            "mũ": ["Hat", "Helmet", "Clothing"]
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
        Lay Objects theo keyframe/vector ordinal 0-based.

        Danh sach JSON duoc natural-sort theo dataset,
        sau do frame_idx duoc dung lam ordinal.

        Khong dung actual video frame_id de doan ten Object JSON.
        """
        cache_key = f"{video_id}_{frame_idx}_{frame_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        v_folder = self._find_video_object_folder(video_id)
        if not v_folder:
            return None
            
        
        json_files = [
            os.path.join(v_folder, name)
            for name in os.listdir(v_folder)
            if name.lower().endswith(".json")
            and os.path.isfile(os.path.join(v_folder, name))
        ]
        json_files.sort(key=natural_sort_key)

        idx = int(frame_idx)
        if idx < 0 or idx >= len(json_files):
            return None

        json_file = json_files[idx]

        
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
        except Exception as exc:
            print(
                "ObjectSearcher: Canh bao khong doc duoc "
                f"{json_file}: {exc}"
            )
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
                # Xu ly loai tru cac tu dong am dac thu cua tieng Viet
                if kw == "đàn" and ("đàn hổ" in text_lower or "đàn dê" in text_lower or "đàn bò" in text_lower or "đàn chim" in text_lower or "người đàn ông" in text_lower or "đàn ong" in text_lower or "đàn cá" in text_lower):
                    if not any(m in text_lower for m in ["chơi đàn", "gảy đàn", "tiếng đàn", "đàn guitar", "đàn piano", "đàn tranh", "nhạc cụ"]):
                        continue
                if kw == "tượng" and ("tượng trưng" in text_lower or "tượng hình" in text_lower):
                    if not any(s in text_lower for s in ["bức tượng", "tượng đài", "tạc tượng", "pho tượng"]):
                        continue
                        
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
            "pineapple", "dish", "spring roll", "robot", "toy", "glass", "tableware", "kitchen utensil", "wheel"
        }
        
        # Bao gom ca cac class da duoc khai bao trong entity_map,
        # tranh hai danh sach OpenImages bi lech nhau.
        for mapped_entities in self.entity_map.values():
            for entity in mapped_entities:
                valid_openimages_classes.add(
                    entity.lower()
                )

        # Match truc tiep ca single-word va multi-word class.
        # Vi du:
        #   "cell phone"
        #   "camera lens"
        #   "space vehicle"
        #   "sports equipment"
        #   "musical instrument"
        for class_name in sorted(
            valid_openimages_classes,
            key=lambda value: (
                -len(value.split()),
                -len(value),
            ),
        ):
            pattern = (
                r"(?<!\w)"
                + re.escape(class_name)
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                text_lower,
                flags=re.IGNORECASE,
            ):
                target_entities.add(class_name)
                
        return sorted(list(target_entities))

    def get_entity_information_weight(self, entity_name, query_text=""):
        """
        Dinh luong trong so thuc the theo nguyen ly Saliency 3 Tang (Visual Saliency Taxonomy):
        - ⭐ Tier 1 (x3.5): Chu the tieu diem doc nhat (Dong vat, Phuong tien, Nhac cu, Mon an cu the, Thiet bi, Nghe thuat...)
        - 🔹 Tier 2 (x2.0): Dao cu, Do noi that, Phu kien ca nhan (Chao, dia, bat, ban, ghe, ke sach, mu, non, kinh, balo, tui...)
        - 🔸 Tier 3 (x0.5): Tac nhan nen & Boi canh pho quat (Person, Man, Woman, Clothing, Shirt, Plant, Tree, Building, Wall...)
        """
        ent = entity_name.lower().strip()
        
        tier3_generic = {
            "person", "man", "woman", "girl", "boy", "human",
            "clothing", "shirt", "dress", "pants", "suit",
            "plant", "tree", "building", "window", "door", "wall", "sky", "floor", "ground", "road", "sand",
            "food", "animal", "vehicle"
        }
        
        tier2_props = {
            "plate", "tableware", "tray", "bowl", "cup", "glass", "kitchen utensil", "bottle", "fork", "knife", "spoon",
            "table", "chair", "bookcase", "book", "bed", "couch", "furniture",
            "hat", "helmet", "backpack", "handbag", "tie", "costume", "box", "bag", "packaging",
            "vase", "clock", "toy"
        }
        
        if ent in tier3_generic:
            return 0.5
        elif ent in tier2_props:
            return 2.0
        else:
            return 3.5

    def extract_tiered_entities(self, query_text):
        """Boc tach thuc the day du 3 Tang: Tier 1 (Chu the), Tier 2 (Dao cu/Noi that), Tier 3 (Boi canh nen)."""
        all_ents = self.extract_target_entities(query_text)
        t1, t2, t3 = [], [], []
        
        for e in all_ents:
            w = self.get_entity_information_weight(e, query_text)
            if w >= 3.0:
                t1.append(e)
            elif w >= 1.5:
                t2.append(e)
            else:
                t3.append(e)
                
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
        has_tier1_target = any(self.get_entity_information_weight(e, query_text) >= 3.0 for e in target_entities)

        
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
                        if ent_lower in target_entities:
                            # Tinh trong so theo Ham luong Thong tin cua thuc the
                            info_weight = self.get_entity_information_weight(ent_lower, query_text)
                            frame_match_score += float(obj["score"]) * (info_weight / 2.0)
                            video_matched_entities.add(ent_lower)
                            if info_weight >= 3.0:
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


