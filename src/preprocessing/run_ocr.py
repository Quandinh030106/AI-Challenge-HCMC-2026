import os
import glob
import json
import torch
from tqdm import tqdm
from src.utils import load_config


try:
    import easyocr
except ImportError:
    easyocr = None

def run_ocr_on_all_keyframes(config_path="configs/default.yaml", step=5):
    """
    Chạy OCR trên tất cả keyframe để trích xuất văn bản trong video.
    Tham số `step`: bước nhảy để chọn frame chạy OCR (mặc định=5 để tránh chạy trùng lặp
    các frame sát nhau gây tốn thời gian, giảm tải xử lý xuống 5 lần).
    """
    if easyocr is None:
        raise ImportError("Vui lòng cài đặt thư viện EasyOCR: pip install easyocr")
        
    # 1. Load cấu hình
    config = load_config(config_path)
    keyframes_dir = config["data"]["keyframes_dir"]
    metadata_dir = config["data"].get("metadata_dir", "data/metadata")
    
    os.makedirs(metadata_dir, exist_ok=True)
    
    # 2. Khởi động EasyOCR Reader (hỗ trợ Tiếng Việt 'vi' và Tiếng Anh 'en')
    use_gpu = torch.cuda.is_available()
    print(f"Khởi tạo EasyOCR (vi + en) trên thiết bị GPU={use_gpu}...")
    reader = easyocr.Reader(['vi', 'en'], gpu=use_gpu)
    
    # 3. Quét tất cả các thư mục chứa keyframe
    video_dirs = [d for d in glob.glob(os.path.join(keyframes_dir, "*")) if os.path.isdir(d)]
    print(f"Tìm thấy {len(video_dirs)} thư mục video để chạy OCR.")
    
    for video_dir in tqdm(video_dirs, desc="Chạy OCR trên video"):
        video_id = os.path.basename(video_dir)
        output_path = os.path.join(metadata_dir, f"{video_id}_ocr.json")
        
        # Bỏ qua nếu video này đã được OCR trước đó
        if os.path.exists(output_path):
            continue
            
        # Lấy tất cả ảnh keyframe
        image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if not image_paths:
            continue
            
        ocr_results = {}
        
        # 4. Duyệt qua từng ảnh theo bước nhảy `step`
        for idx in range(0, len(image_paths), step):
            img_path = image_paths[idx]
            frame_id = os.path.splitext(os.path.basename(img_path))[0]
            
            try:
                # Chạy OCR
                results = reader.readtext(img_path)
                # Ghép các đoạn text nhận diện được có độ tin cậy > 0.3
                texts = [res[1] for res in results if res[2] > 0.3]
                if texts:
                    ocr_results[frame_id] = " ".join(texts)
            except Exception as e:
                print(f"Lỗi chạy OCR trên ảnh {img_path}: {e}")
                
        # 5. Lưu kết quả OCR của video
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            
    print("Quá trình chạy OCR trên toàn bộ keyframes hoàn thành!")

if __name__ == "__main__":
    run_ocr_on_all_keyframes()

