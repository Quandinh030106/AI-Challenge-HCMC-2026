import os
import sys
import json
import re
import subprocess

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    try:
        print("SparseSearcher: Dang tu dong cai dat thu vien 'rank-bm25'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25", "--quiet"])
        from rank_bm25 import BM25Okapi
        print("SparseSearcher: Cai dat 'rank-bm25' thanh cong!")
    except Exception as e:
        print(f"SparseSearcher: Canh bao ({e}). BM25Okapi khong kha thi.")
        BM25Okapi = None

class SparseSearcher:

    def __init__(self, config):
        self.config = config
        self.metadata_dir = config["data"]["metadata_dir"]
        self.corpus = []
        self.video_ids = []
        self.bm25 = None
        self.build_index()
        
    def preprocess_text(self, text):
        """Tien xu ly van ban: viet thuong va tach tu co ban, loai bo stopwords pho bien."""
        if not text:
            return []
        raw_words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {
            "đoạn", "video", "clip", "về", "một", "chương", "trình", "của", "tên", "là", 
            "trong", "có", "thể", "thấy", "này", "đang", "đi", "tại", "thuộc", "tỉnh", 
            "hãy", "tìm", "chính", "xác", "phân", "cảnh", "bắt", "đầu", "với", "hình", 
            "ảnh", "sau", "đó", "tiếp", "theo", "người", "cần", "biết", "gồm", "các"
        }
        filtered = [w for w in raw_words if w not in stopwords and len(w) > 1]
        return filtered if filtered else raw_words
        
    def extract_text_from_obj(self, obj):
        """Boc tach de quy tat ca cac chuoi van ban trong JSON."""
        text = ""
        if isinstance(obj, str):
            text += " " + obj
        elif isinstance(obj, list):
            for item in obj:
                text += self.extract_text_from_obj(item)
        elif isinstance(obj, dict):
            for val in obj.values():
                text += self.extract_text_from_obj(val)
        return text

    def build_index(self):
        """Doc file metadata/OCR de dung chi muc BM25."""
        print(f"SparseSearcher: Dang xay dung chi muc BM25 tu: {self.metadata_dir}")
        json_files = []
        scan_dir = self.metadata_dir if os.path.exists(self.metadata_dir) else "/kaggle/input"

        if os.path.exists(scan_dir):
            for root, _, files in os.walk(scan_dir):
                root_lower = root.lower()
                if "object" in root_lower or "keyframe" in root_lower or "video" in root_lower:
                    continue
                for file in files:
                    file_lower = file.lower()
                    if file_lower.endswith(".json") and file != "local_val_gt.json":
                        json_files.append(os.path.join(root, file))
        
        print(f"SparseSearcher: Tim thay {len(json_files)} file JSON metadata.")
        video_texts = {}
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            if filename == "local_val_gt.json":
                continue
                
            video_id = filename.split('_ocr')[0].split('.')[0]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text_content = self.extract_text_from_obj(data)
                
                if video_id not in video_texts:
                    video_texts[video_id] = ""
                video_texts[video_id] += " " + text_content
            except Exception:
                pass
                
        for video_id, text in video_texts.items():
            tokenized_doc = self.preprocess_text(text)
            if tokenized_doc:
                self.corpus.append(tokenized_doc)
                self.video_ids.append(video_id)
                
        if self.corpus and BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"SparseSearcher: Da hoan thanh chi muc BM25 cho {len(self.video_ids)} videos.")
        else:
            print("SparseSearcher: Canh bao: Khong the khoi tao BM25 (Co the do thieu thu vien rank-bm25 hoac khong co metadata).")


    def search(self, query_text, top_k_videos=25):
        """Tim kiem video theo tu khoa cau hoi, tang cuong trong so cho Named Entities."""
        if self.bm25 is None or not self.video_ids:
            return []
            
        tokenized_query = self.preprocess_text(query_text)
        
        # Phat hien cac thuc the viet hoa va tu khoa hiem de nhan trong so (Entity Boosting 5x)
        entities = re.findall(r'\b[A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zđáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+\b', query_text)
        rare_keywords = ["fana", "khánh hòa", "đền thờ nguyễn trung trực", "đền thờ", "đình thần", "nguyễn trung trực", "kiên giang", "lausanne", "spielberg", "covid", "panna cotta", "múa lân", "gỏi cuốn"]

        
        boosted_tokens = list(tokenized_query)
        for ent in entities:
            ent_lower = ent.lower()
            if len(ent_lower) > 2:
                boosted_tokens.extend([ent_lower] * 4)
                
        for rk in rare_keywords:
            if rk in query_text.lower():
                for kw in rk.split():
                    boosted_tokens.extend([kw] * 4)
                    
        scores = self.bm25.get_scores(boosted_tokens)
        
        results = []
        for i, video_id in enumerate(self.video_ids):
            score = float(scores[i])
            if score > 0:
                results.append({"video_id": video_id, "sparse_score": score})
                
        results.sort(key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k_videos]

