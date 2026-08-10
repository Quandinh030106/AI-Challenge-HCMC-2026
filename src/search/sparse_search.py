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
        
    def build_index(self):
        """Đọc toàn bộ file metadata và OCR của các video để dựng chỉ mục BM25."""
        print(f"SparseSearcher: Đang xây dựng chỉ mục BM25 từ thư mục: {self.metadata_dir}")
        
        # Quét tất cả các file .json và .JSON trong thư mục metadata (bao gồm thư mục con)
        json_files = (
            glob.glob(os.path.join(self.metadata_dir, "**/*.json"), recursive=True) +
            glob.glob(os.path.join(self.metadata_dir, "**/*.JSON"), recursive=True)
        )
        
        print(f"SparseSearcher: Tìm thấy {len(json_files)} file JSON trong thư mục metadata.")
        if json_files:
            print(f"SparseSearcher: 3 file JSON đầu tiên tìm được: {json_files[:3]}")
            
        # Gom dữ liệu văn bản theo từng video_id
        video_texts = {}
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            # Bỏ qua file Ground Truth của local validation
            if filename == "local_val_gt.json":
                continue
                
            # Xác định video_id từ tên file (ví dụ: L01_V001.json hoặc L01_V001_ocr.json)
            video_id = filename.split('_ocr')[0].split('.')[0]
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                text_content = ""
                if isinstance(data, dict):
                    # Nếu là file OCR dạng {"frame_id": "text"}
                    for val in data.values():
                        if isinstance(val, str):
                            text_content += " " + val
                    # Nếu là file Youtube metadata có title, description
                    if "title" in data:
                        text_content += " " + str(data["title"])
                    if "description" in data:
                        text_content += " " + str(data["description"])
                
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
            print("SparseSearcher: Cảnh báo: Không có văn bản nào được index cho BM25! (Có thể thư mục chứa metadata rỗng hoặc không đúng cấu trúc)")

            
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
