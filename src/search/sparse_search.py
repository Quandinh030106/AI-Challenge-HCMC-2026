import os
import json
import re
from rank_bm25 import BM25Okapi

class SparseSearcher:
    def __init__(self, config):
        self.config = config
        self.metadata_dir = config["data"]["metadata_dir"]
        self.corpus = []
        self.video_ids = []
        self.bm25 = None
        self.build_index()
        
    def preprocess_text(self, text):
        """Tien xu ly van ban: viet thuong va tach tu co ban."""
        if not text:
            return []
        return re.findall(r'\b\w+\b', text.lower())
        
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
        
        if os.path.exists(self.metadata_dir):
            for root, _, files in os.walk(self.metadata_dir):
                for file in files:
                    if file.lower().endswith(".json") and file != "local_val_gt.json":
                        json_files.append(os.path.join(root, file))
        
        if not json_files:
            parent_dir = os.path.dirname(self.metadata_dir)
            if os.path.exists(parent_dir):
                for root, _, files in os.walk(parent_dir):
                    if "metadata" in root.lower() or "media-info" in root.lower():
                        for file in files:
                            if file.lower().endswith(".json") and file != "local_val_gt.json":
                                json_files.append(os.path.join(root, file))
        
        print(f"SparseSearcher: Tim thay {len(json_files)} file JSON.")
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
                
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"SparseSearcher: Da hoan thanh chi muc BM25 cho {len(self.video_ids)} videos.")
        else:
            print("SparseSearcher: Canh bao: Khong co van ban nao duoc index cho BM25!")

    def search(self, query_text, top_k_videos=10):
        """Tim kiem video theo tu khoa cau hoi, tra ve diem so BM25."""
        if self.bm25 is None or not self.video_ids:
            return []
            
        tokenized_query = self.preprocess_text(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for i, video_id in enumerate(self.video_ids):
            score = float(scores[i])
            if score > 0:
                results.append({"video_id": video_id, "sparse_score": score})
                
        results.sort(key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k_videos]
