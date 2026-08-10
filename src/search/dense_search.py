import os
import glob
import numpy as np
import torch
from transformers import (
    CLIPModel, CLIPProcessor, 
    SiglipModel, SiglipProcessor, 
    AutoModel, AutoProcessor
)

class DenseSearcher:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config["models"]["clip_model"]
        self.features_dir = config["data"]["features_dir"]
        
        print(f"DenseSearcher: Đang khởi tạo mô hình {self.model_name}...")
        
        # Sử dụng đúng class chuyên biệt để get_text_features và get_image_features hoạt động chuẩn xác
        if "siglip" in self.model_name.lower():
            self.processor = SiglipProcessor.from_pretrained(self.model_name)
            self.model = SiglipModel.from_pretrained(self.model_name).to(self.device)
        elif "clip" in self.model_name.lower():
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
        self.model.eval()
        
    def encode_text(self, text):
        """Mã hóa văn bản truy vấn thành vector đặc trưng và chuẩn hóa L2."""
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs)
            
            # Xử lý tương thích với các phiên bản khác nhau của thư viện transformers
            if isinstance(text_outputs, torch.Tensor):
                text_features = text_outputs
            elif hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
                
            # Chuẩn hóa L2-norm
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]


            
    def search(self, query_text, top_k_videos=10):
        """Tìm kiếm video có độ tương đồng cosine lớn nhất với câu truy vấn."""
        query_vector = self.encode_text(query_text)
        
        # Quét tất cả các file vector .npy trong thư mục features (kể cả thư mục con)
        feature_files = glob.glob(os.path.join(self.features_dir, "**/*.npy"), recursive=True)
        if not feature_files:
            feature_files = glob.glob(os.path.join(self.features_dir, "*.npy"))
            
        if not feature_files:
            print(f"Cảnh báo: Không tìm thấy file .npy nào trong thư mục {self.features_dir}")
            return []
            
        results = []
        for file_path in feature_files:
            video_id = os.path.splitext(os.path.basename(file_path))[0]
            try:
                # Load ma trận vector đặc trưng của video
                video_features = np.load(file_path) # Shape: [num_frames, feature_dim]
                
                # Tính Cosine Similarity bằng Dot Product
                scores = np.dot(video_features, query_vector) # Shape: [num_frames]
                
                # Tìm frame có độ tương đồng lớn nhất
                max_score_idx = np.argmax(scores)
                max_score = scores[max_score_idx]
                
                results.append({
                    "video_id": video_id,
                    "max_score": float(max_score),
                    "best_frame_idx": int(max_score_idx),
                    "all_scores": scores
                })
            except Exception as e:
                print(f"DenseSearcher: Lỗi xử lý file {file_path}: {e}")
                
        # Sắp xếp kết quả theo điểm tương đồng giảm dần
        results = sorted(results, key=lambda x: x["max_score"], reverse=True)
        return results[:top_k_videos]
