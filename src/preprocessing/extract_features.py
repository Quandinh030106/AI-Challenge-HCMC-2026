import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from src.utils import load_config

def extract_all_features(config_path="configs/default.yaml", batch_size=32):
    # 1. Load cấu hình đường dẫn
    config = load_config(config_path)
    keyframes_dir = config["data"]["keyframes_dir"]
    features_dir = config["data"]["features_dir"]
    model_name = config["models"]["clip_model"]
    
    os.makedirs(features_dir, exist_ok=True)
    
    # 2. Khởi tạo mô hình SigLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang khởi tạo SigLIP: {model_name} trên thiết bị: {device}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 3. Lấy tất cả các thư mục chứa keyframe (mỗi thư mục tương ứng với một video_id)
    video_dirs = [d for d in glob.glob(os.path.join(keyframes_dir, "*")) if os.path.isdir(d)]
    print(f"Tìm thấy {len(video_dirs)} thư mục keyframe video cần xử lý.")
    
    for video_dir in tqdm(video_dirs, desc="Trích xuất đặc trưng"):
        video_id = os.path.basename(video_dir)
        output_path = os.path.join(features_dir, f"{video_id}.npy")
        
        # Bỏ qua nếu đã trích xuất trước đó
        if os.path.exists(output_path):
            continue
            
        # Lấy và sắp xếp thứ tự các keyframe
        image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if not image_paths:
            print(f"Cảnh báo: Không tìm thấy ảnh trong thư mục {video_dir}")
            continue
            
        video_features = []
        
        # 4. Xử lý ảnh theo từng Batch để tối ưu tài nguyên GPU
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    print(f"Lỗi đọc ảnh {img_path}: {e}")
                    
            if not batch_images:
                continue
                
            # Đưa qua bộ tiền xử lý và trích xuất vector đặc trưng
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                # Chuẩn hóa L2-norm để có thể tính Cosine Similarity trực tiếp bằng Dot Product
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                features = outputs.cpu().numpy()
                video_features.append(features)
                
        if video_features:
            # Ghép nối các batch lại thành 1 tensor dạng [num_frames, feature_dim]
            video_features = np.concatenate(video_features, axis=0)
            np.save(output_path, video_features)
            
    print("Quá trình trích xuất đặc trưng ảnh SigLIP hoàn thành!")

if __name__ == "__main__":
    extract_all_features()


