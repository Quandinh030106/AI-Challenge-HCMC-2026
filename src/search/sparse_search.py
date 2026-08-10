import os
import glob
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
        """Tiền xử lý văn bản: viết thường và tách từ cơ bản."""
        if not text:
            return []
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
        
    def extract_text_from_obj(self, obj):
        """Bóc tách đệ quy tất cả các chuỗi văn bản trong file JSON (kể cả trong List hay Dict)."""
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
        """Đọc toàn bộ file metadata và OCR của các video để dựng chỉ mục BM25."""
        print(f"SparseSearcher: Đang xây dựng chỉ mục BM25 từ thư mục: {self.metadata_dir}")
        
        json_files = []
        # Thử 1: Quét trong thư mục metadata_dir được cấu hình
        if os.path.exists(self.metadata_dir):
            for root, _, files in os.walk(self.metadata_dir):
                for file in files:
                    if file.lower().endswith(".json") and file != "local_val_gt.json":
                        json_files.append(os.path.join(root, file))
        
        # Thử 2 (Fallback): Nếu không tìm thấy, tự động quét thư mục cha (ví dụ /kaggle/input)
        if not json_files:
            parent_dir = os.path.dirname(self.metadata_dir)
            if os.path.exists(parent_dir):
                print(f"SparseSearcher: Không thấy JSON trong {self.metadata_dir}. Đang tự động quét trong {parent_dir}...")
                for root, _, files in os.walk(parent_dir):
                    if "metadata" in root.lower() or "media-info" in root.lower():
                        for file in files:
                            if file.lower().endswith(".json") and file != "local_val_gt.json":
                                json_files.append(os.path.join(root, file))
        
        print(f"SparseSearcher: Tìm thấy {len(json_files)} file JSON trong hệ thống.")
        if json_files:
            print(f"SparseSearcher: Ví dụ 3 file JSON tìm thấy: {[os.path.basename(f) for f in json_files[:3]]}")

            
        # Gom dữ liệu văn bản theo từng video_id
        video_texts = {}
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            if filename == "local_val_gt.json":
                continue
                
            # Xác định video_id từ tên file (ví dụ: L21_V001.json -> L21_V001)
            video_id = filename.split('_ocr')[0].split('.')[0]
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Trích xuất toàn bộ chữ từ file JSON (title, description, tags, keywords...)
                text_content = self.extract_text_from_obj(data)
                
                if video_id not in video_texts:
                    video_texts[video_id] = ""
                video_texts[video_id] += " " + text_content
            except Exception as e:
                pass
                
        # Tạo corpus để train BM25
        for video_id, text in video_texts.items():
            tokenized_doc = self.preprocess_text(text)
            if tokenized_doc:
                self.corpus.append(tokenized_doc)
                self.video_ids.append(video_id)
                
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            print(f"SparseSearcher: Đã hoàn thành chỉ mục BM25 cho {len(self.video_ids)} videos.")
        else:
            print("SparseSearcher: Cảnh báo: Không có văn bản nào được index cho BM25!")


            
    def search(self, query_text, top_k_videos=10):
        """Tìm kiếm video theo từ khóa câu hỏi, trả về điểm số BM25."""
        if self.bm25 is None or not self.video_ids:
            return []
            
        tokenized_query = self.preprocess_text(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for i, video_id in enumerate(self.video_ids):
            score = float(scores[i])
            if score > 0: # Chỉ giữ các video có khớp từ khóa ít nhất 1 từ
                results.append({
                    "video_id": video_id,
                    "sparse_score": score
                })
                
        # Sắp xếp kết quả giảm dần theo điểm BM25
        results = sorted(results, key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k_videos]
