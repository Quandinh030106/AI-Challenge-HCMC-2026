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
        
        print(f"DenseSearcher: Khoi tao mo hinh {self.model_name} tren {self.device}...")
        
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
        
        self.global_tensor = None
        self.video_metadata_list = []
        self.all_video_ids = []
        self.video_features_dict = {}
        
        self.load_and_build_global_matrix()

    def load_and_build_global_matrix(self):
        """Nap toan bo vector dac trung vao GPU thanh mot ma tran duy nhat."""
        print(f"DenseSearcher: Dang nap vector tu thu muc: {self.features_dir}")
        feature_files = []
        
        # 1. Uu tien doc truc tiep tu duong dan features_dir duoc cau hinh (sieu nhanh < 0.01s)
        if self.features_dir and os.path.exists(self.features_dir):
            for root, _, files in os.walk(self.features_dir):
                for file in files:
                    if file.lower().endswith(".npy"):
                        feature_files.append(os.path.join(root, file))

        # 2. Fallback: Neu chua tim thay, moi quet trong /kaggle/input
        if not feature_files and os.path.exists("/kaggle/input"):
            for root, _, files in os.walk("/kaggle/input"):
                root_lower = root.lower()
                if "keyframe" in root_lower or "video" in root_lower:
                    continue
                for file in files:
                    if file.lower().endswith(".npy"):
                        feature_files.append(os.path.join(root, file))

        feature_files = sorted(feature_files)
        if not feature_files:
            print("DenseSearcher: Canh bao: Khong tim thay file .npy nao!")
            return

        print(f"DenseSearcher: Tim thay {len(feature_files)} file .npy.")

        all_vectors = []
        current_idx = 0
        
        for file_path in feature_files:
            video_id = os.path.splitext(os.path.basename(file_path))[0]
            try:
                feats = np.load(file_path)
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
            except Exception:
                pass
                
        if all_vectors:
            concat_matrix = np.vstack(all_vectors)
            if self.device == "cuda":
                self.global_tensor = torch.from_numpy(concat_matrix).half().to(self.device)
            else:
                self.global_tensor = torch.from_numpy(concat_matrix).float()
            print(f"DenseSearcher: Da nap ma tran toan cuc {self.global_tensor.shape} ({len(self.all_video_ids)} videos) vao {self.device}.")

    def encode_text_matrix(self, text_or_list):
        """Ma hoa tap prompt ensemble thanh ma tran cac vector (M, 512) da chuan hoa L2."""
        text_inputs = [text_or_list] if isinstance(text_or_list, str) else list(text_or_list)
        inputs = self.processor(
            text=text_inputs, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(self.device)
        
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
            return text_features.half() if self.device == "cuda" else text_features.float()

    def encode_text(self, text_or_list):
        """Ma hoa don le hoac trung binh vector dac trung."""
        mat = self.encode_text_matrix(text_or_list)
        if mat.shape[0] > 1:
            mean_vec = mat.mean(dim=0, keepdim=True)
            return mean_vec / mean_vec.norm(p=2, dim=-1, keepdim=True)
        return mat

    def search(self, query_text_or_ensemble, top_k_videos=100):
        """Tim kiem video bang phep nhan ma tran GPU da Prompt (Multi-Prompt Fusion)."""
        if self.global_tensor is None:
            return []

        q_matrix = self.encode_text_matrix(query_text_or_ensemble)
        
        with torch.no_grad():
            # global_tensor: (N_frames, 512), q_matrix: (M_prompts, 512)
            # sim_matrix: (N_frames, M_prompts)
            sim_matrix = torch.matmul(self.global_tensor, q_matrix.T)
            
            if q_matrix.shape[0] > 1:
                # Ket hop 50% Max Peak (bat chi tiet dac biet) va 50% Mean (bao toan ngu canh lon)
                max_scores, _ = torch.max(sim_matrix, dim=-1)
                mean_scores = torch.mean(sim_matrix, dim=-1)
                sim_scores = 0.5 * max_scores + 0.5 * mean_scores
            else:
                sim_scores = sim_matrix.squeeze(-1)
                
            sim_scores_np = sim_scores.float().cpu().numpy()

        results = []
        for meta in self.video_metadata_list:
            video_id = meta["video_id"]
            start_i = meta["start_idx"]
            end_i = meta["end_idx"]
            
            v_scores = sim_scores_np[start_i:end_i]
            if len(v_scores) == 0:
                continue
                
            if len(v_scores) >= 3:
                # Lam min chuoi diem bang Gaussian Kernel (sigma=1.2) de loc nhieu
                kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
                kernel /= kernel.sum()
                smoothed = np.convolve(v_scores, kernel, mode='same')
                max_idx = int(np.argmax(smoothed))
                # Diem tong hop ket hop giua do duy tri cua phan canh va do net cua frame dinh
                max_score = float(0.6 * smoothed[max_idx] + 0.4 * v_scores[max_idx])
            else:
                max_idx = int(np.argmax(v_scores))
                max_score = float(v_scores[max_idx])
            
            results.append({
                "video_id": video_id,
                "max_score": max_score,
                "best_frame_idx": int(max_idx),
                "all_scores": v_scores
            })

        results.sort(key=lambda x: x["max_score"], reverse=True)
        self.last_dense_dict = {r["video_id"]: r for r in results}
        return results if top_k_videos is None else results[:top_k_videos]



