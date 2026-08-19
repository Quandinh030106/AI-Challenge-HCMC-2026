import os
import glob
import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.tasks.task1_kis import get_frame_id_from_idx

_vlm_model = None
_vlm_processor = None

def load_vlm(model_id="Qwen/Qwen2-VL-7B-Instruct"):
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        print(f"VQA: Dang load mo hinh Qwen2-VL {model_id}...")
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        _vlm_processor = AutoProcessor.from_pretrained(model_id)
        print("VQA: Mo hinh VLM da duoc load thanh cong.")
    return _vlm_model, _vlm_processor

def solve_task2(query_text, question, fused_candidates, keyframes_dir, model_id="Qwen/Qwen2-VL-7B-Instruct"):
    """Giai quyet Task 2 (Visual Q&A)."""
    if not fused_candidates:
        return {"video_id": "none", "frame_id": "0000", "answer": "không rõ"}
        
    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    dense_info = best_candidate.get("dense_info")
    
    if dense_info is not None and "all_scores" in dense_info:
        scores = dense_info["all_scores"]
        best_frame_idx = int(np.argmax(scores))
    else:
        best_frame_idx = 0
        
    frame_id = get_frame_id_from_idx(keyframes_dir, video_id, best_frame_idx)
    
    # Tim truc tiep file anh vat ly bang Direct Candidate Path (sieu toc < 0.001s, khong dung recursive scan)
    level = video_id.split('_')[0] if '_' in video_id else ""
    fid_str = f"{int(frame_id):04d}" if str(frame_id).isdigit() else str(frame_id)
    
    candidate_img_paths = [
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{fid_str}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", video_id, f"{fid_str}.jpg"),
        os.path.join(keyframes_dir, level, "keyframes", video_id, f"{fid_str}.jpg"),
        os.path.join(keyframes_dir, "keyframes", video_id, f"{fid_str}.jpg"),
        os.path.join(keyframes_dir, video_id, f"{fid_str}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{frame_id}.jpg"),
        os.path.join(keyframes_dir, video_id, f"{frame_id}.jpg")
    ]
    
    image_path = None
    for p in candidate_img_paths:
        if os.path.exists(p):
            image_path = p
            break
            
    if not image_path:
        # Neu chua co anh vat ly (vi du tren tap test chi co .npy), tra ve dap an an toan
        return {"video_id": video_id, "frame_id": frame_id, "answer": "không rõ"}
        
    model, processor = load_vlm(model_id)

    
    # Prompt toi uu cho Task 2 VQA: ep model tra ve truc dien tu khoa (Mau sac, So luong, Ten vat the)
    prompt_text = (
        f"Dựa vào bức ảnh này, hãy trả lời câu hỏi sau bằng Tiếng Việt một cách ngắn gọn, trực diện nhất "
        f"(chỉ trả lời 1 đến 3 từ, chỉ nêu đúng từ khóa đáp án như màu sắc, con số, hoặc tên vật thể; "
        f"không giải thích, không dùng câu dẫn dài dòng): '{question}'"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=25)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
    raw_answer = output_text[0].strip()
    
    # Loc bo cac tien to va tu dem thua (vi du: "Dap an: ", "La ", "Do la ")
    clean_ans = raw_answer
    for prefix in ["đáp án:", "đáp án là:", "trả lời:", "câu trả lời:", "là", "đó là", "nó là", "có"]:
        if clean_ans.lower().startswith(prefix):
            clean_ans = clean_ans[len(prefix):].strip()
            
    clean_ans = clean_ans.rstrip('.!?,;:')
    if clean_ans:
        clean_ans = clean_ans[0].upper() + clean_ans[1:]
    else:
        clean_ans = "Không rõ"
        
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "answer": clean_ans
    }

