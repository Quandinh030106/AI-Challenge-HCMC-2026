import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor, AutoModel, AutoProcessor
from src.utils import load_config

def extract_all_features(config_path="configs/default.yaml", batch_size=32):
    """Trich xuat vector dac trung anh theo batch tren GPU."""
    config = load_config(config_path)
    keyframes_dir = config["data"]["keyframes_dir"]
    features_dir = config["data"]["features_dir"]
    model_name = config["models"]["clip_model"]
    
    os.makedirs(features_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Khoi tao model: {model_name} tren {device}...")
    
    if "siglip" in model_name.lower():
        processor = SiglipProcessor.from_pretrained(model_name)
        model = SiglipModel.from_pretrained(model_name).to(device)
    elif "clip" in model_name.lower():
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
    model.eval()
    video_dirs = [d for d in glob.glob(os.path.join(keyframes_dir, "*")) if os.path.isdir(d)]
    print(f"Tim thay {len(video_dirs)} thu muc keyframe can xu ly.")
    
    for video_dir in tqdm(video_dirs, desc="Trich xuat dac trung"):
        video_id = os.path.basename(video_dir)
        output_path = os.path.join(features_dir, f"{video_id}.npy")
        
        if os.path.exists(output_path):
            continue
            
        image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if not image_paths:
            continue
            
        video_features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                except Exception:
                    pass
                    
            if not batch_images:
                continue
                
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.image_embeds if hasattr(outputs, "image_embeds") else (outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0])
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                video_features.append(outputs.cpu().numpy())
                
        if video_features:
            np.save(output_path, np.concatenate(video_features, axis=0))
            
    print("Hoan thanh trich xuat dac trung.")

if __name__ == "__main__":
    extract_all_features()
