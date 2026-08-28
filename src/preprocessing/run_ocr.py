import os
import glob
import json
import torch
from tqdm import tqdm
from src.utils import load_config
from src.utils import natural_sort_key

try:
    import easyocr
except ImportError:
    easyocr = None

def run_ocr_on_all_keyframes(config_path="configs/default.yaml", step=5):
    """Trich xuat chu tu keyframes bang EasyOCR (buoc nhay step=5)."""
    if easyocr is None:
        raise ImportError("Vui long cai dat easyocr: pip install easyocr")
        
    config = load_config(config_path)
    keyframes_dir = config["data"]["keyframes_dir"]
    metadata_dir = config["data"].get("metadata_dir", "data/metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    use_gpu = torch.cuda.is_available()
    print(f"Khoi tao EasyOCR (vi + en) tren GPU={use_gpu}...")
    reader = easyocr.Reader(['vi', 'en'], gpu=use_gpu)
    
    video_dirs = [d for d in glob.glob(os.path.join(keyframes_dir, "*")) if os.path.isdir(d)]
    print(f"Tim thay {len(video_dirs)} thu muc video de chay OCR.")
    
    for video_dir in tqdm(video_dirs, desc="Chay OCR"):
        video_id = os.path.basename(video_dir)
        output_path = os.path.join(metadata_dir, f"{video_id}_ocr.json")
        
        if os.path.exists(output_path):
            continue
            
        image_paths = sorted(
            glob.glob(os.path.join(video_dir, "*.jpg")),
            key=natural_sort_key,
        )
        if not image_paths:
            continue
            
        ocr_results = {}
        for idx in range(0, len(image_paths), step):
            img_path = image_paths[idx]
            frame_id = os.path.splitext(os.path.basename(img_path))[0]
            try:
                results = reader.readtext(img_path)
                texts = [res[1] for res in results if res[2] > 0.3]
                if texts:
                    ocr_results[frame_id] = " ".join(texts)
            except Exception:
                pass
                
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            
    print("Hoan thanh chay OCR.")

if __name__ == "__main__":
    run_ocr_on_all_keyframes()
