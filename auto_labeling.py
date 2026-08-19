import os
import json
import torch
from glob import glob
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def setup_model():
    print("Đang khởi tạo Qwen2-VL-7B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor

def generate_caption(model, processor, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Mô tả chi tiết hành động, bối cảnh và các đối tượng chính trong hình ảnh này bằng tiếng Việt. Tối đa 2 câu."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=60)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def main():
    # Thư mục keyframes trên Kaggle
    keyframes_dir = "/kaggle/input/ai-challenge-hcmc-2026-keyframes"
    output_file = "auto_labels.json"
    
    model, processor = setup_model()
    image_paths = glob(os.path.join(keyframes_dir, "**/*.jpg"), recursive=True)[:100] # Test trước 100 ảnh
    
    results = {}
    for img_path in tqdm(image_paths, desc="Đang gán nhãn"):
        try:
            results[os.path.basename(img_path)] = generate_caption(model, processor, img_path)
        except Exception as e:
            print(f"Lỗi: {e}")
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
