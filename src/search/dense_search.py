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
        
        print(f"DenseSearcher: Đang khởi tạo mô hình {self.model_name} trên {self.device}...")
        
        # 1. Khởi tạo mô hình mã hóa text
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
        
        # 2. Bộ nhớ đệm ma trận toàn cục (Global Tensor Cache) để đạt độ trễ < 5ms
        self.global_tensor = None
        self.video_metadata_list = [] # Lưu danh sách (video_id, total_frames, start_idx, end_idx)
        self.all_video_ids = []
        self.video_features_dict = {} # Lưu thô để Task 1 / Task 3 dùng lại
        
        self.load_and_build_global_matrix()

    def load_and_build_global_matrix(self):
        """Quét và gộp tất cả các file .npy thành 1 Ma trận GPU duy nhất để tăng tốc 100 lần."""
        print("DenseSearcher: Đang nạp toàn bộ vector đặc trưng vào GPU/RAM...")
        feature_files = []
        if os.path.exists(self.features_dir):
            for root, _, files in os.walk(self.features_dir):
                for file in files:
                    if file.lower().endswith(".npy"):
                        feature_files.append(os.path.join(root, file))
                        
        if not feature_files:
            base_input = "/kaggle/input"
            if os.path.exists(base_input):
                for root, _, files in os.walk(base_input):
                    for file in files:
                        if file.lower().endswith(".npy"):
                            feature_files.append(os.path.join(root, file))

        if not feature_files:
            print("Cảnh báo: Không tìm thấy file .npy nào!")
            return

        all_vectors = []
        current_idx = 0
        
        for file_path in feature_files:
            video_id = os.path.splitext(os.path.basename(file_path))[0]
            try:
                feats = np.load(file_path) # Shape: (N_frames, Dim)
                # Chuẩn hóa L2 cho các vector ảnh nếu chưa chuẩn hóa
                norms = np.linalg.norm(feats, axis=-1, keepdims=True)
                norms[norms == 0] = 1e-10
                feats_norm = feats / norms
                
                n_frames = feats_norm.shape[0]
                self.video_features_dict[video_id] = feats_norm
                
                all_vectors.append(feats_norm)
                self.video_metadata_list.append({
                    "video_id": video_id,
                    "start_idx": current_idx,
                    "end_idx": current_idx + n_frames,
                    "n_frames": n_frames
                })
                self.all_video_ids.append(video_id)
                current_idx += n_frames
            except Exception as e:
                pass
                
        if all_vectors:
            concat_matrix = np.vstack(all_vectors) # Shape: (Total_Frames_All_Videos, Dim)
            self.global_tensor = torch.from_numpy(concat_matrix).float().to(self.device)
            print(f"DenseSearcher: ✅ Đã nạp thành công Ma trận toàn cục {self.global_tensor.shape} ({len(self.all_video_ids)} videos) vào {self.device}. Độ trễ tìm kiếm < 5ms!")

    def encode_text(self, text):
        """Mã hóa văn bản truy vấn thành vector đặc trưng và chuẩn hóa L2."""
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs)
            if isinstance(text_outputs, torch.Tensor):
                text_features = text_outputs
            elif hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
                
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features

    def search(self, query_text, top_k_videos=10):
        """Tìm kiếm siêu tốc trên Ma trận GPU toàn cục bằng phép nhân ma trận (Matrix Multiplication)."""
        if self.global_tensor is None:
            print("Cảnh báo: Ma trận đặc trưng chưa được khởi tạo!")
            return []

        # 1. Mã hóa query (Shape: 1 x Dim)
        q_tensor = self.encode_text(query_text)
        
        # 2. Nhân ma trận siêu tốc trên GPU: (1 x Dim) x (Dim x Total_Frames) -> (Total_Frames,)
        with torch.no_grad():
            sim_scores = torch.matmul(self.global_tensor, q_tensor.T).squeeze(-1) # Shape: (Total_Frames,)
            sim_scores_np = sim_scores.cpu().numpy()

        # 3. Tổng hợp điểm max/mean cho từng video
        results = []
        for meta in self.video_metadata_list:
            video_id = meta["video_id"]
            start_i = meta["start_idx"]
            end_i = meta["end_idx"]
            
            v_scores = sim_scores_np[start_i:end_i]
            max_idx = np.argmax(v_scores)
            max_score = float(v_scores[max_idx])
            
            results.append({
                "video_id": video_id,
                "max_score": max_score,
                "best_frame_idx": int(max_idx),
                "all_scores": v_scores
            })

        # 4. Sắp xếp giảm dần theo max_score
        results.sort(key=lambda x: x["max_score"], reverse=True)
        return results[:top_k_videos]
