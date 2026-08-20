import os
import glob
import json
import re
import torch
import numpy as np

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from src.tasks.task1_kis import get_frame_id_from_idx

_vlm_model = None
_vlm_processor = None

def load_vlm(model_id="Qwen/Qwen2-VL-2B-Instruct"):
    """Nap mo hinh VLM dung chung cho VQA va Visual Re-ranking sieu nhe chong OOM."""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        print(f"VQA: Nap mo hinh {model_id}...")
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        try:
            from transformers import Qwen2VLForConditionalGeneration
            _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
        except Exception:
            from transformers import AutoModelForVision2Seq
            _vlm_model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        try:
            _vlm_processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        except Exception:
            _vlm_processor = AutoProcessor.from_pretrained(model_id)
        print("VQA: Khoi tao mo hinh VLM thanh cong (toi uu VRAM).")
    return _vlm_model, _vlm_processor

def find_image_for_frame(keyframes_dir, video_id, frame_idx, frame_id=""):
    """Dinh vi nhanh file anh vat ly cua mot frame idx."""
    level = video_id.split('_')[0] if '_' in video_id else ""
    idx_4d = f"{frame_idx:04d}"
    idx_5d = f"{frame_idx:05d}"
    idx_raw = str(frame_idx)
    idx_1based = f"{frame_idx + 1:04d}"
    fid_str = f"{int(frame_id):04d}" if str(frame_id).isdigit() and frame_id else str(frame_id)
    
    candidate_img_names = [idx_4d, idx_5d, idx_raw, idx_1based]
    if fid_str:
        candidate_img_names.append(fid_str)
        
    for name in candidate_img_names:
        candidate_paths = [
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, level, "keyframes", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, "keyframes", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{name}.jpeg"),
            os.path.join(keyframes_dir, video_id, f"{name}.png")
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return p
                
    # Fallback doc thu muc
    folder_candidates = [
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id),
        os.path.join(keyframes_dir, f"Keyframes_{level}", video_id),
        os.path.join(keyframes_dir, level, "keyframes", video_id),
        os.path.join(keyframes_dir, "keyframes", video_id),
        os.path.join(keyframes_dir, video_id)
    ]
    for fc in folder_candidates:
        if os.path.exists(fc):
            all_imgs = sorted(glob.glob(os.path.join(fc, "*.jpg")) + glob.glob(os.path.join(fc, "*.jpeg")) + glob.glob(os.path.join(fc, "*.png")))
            if all_imgs:
                target_idx = min(frame_idx, len(all_imgs) - 1)
                return all_imgs[target_idx]
    return None

def clean_vlm_answer(raw_answer):
    """Lam sach tien to dan chuyen nhung GIU NGUYEN VEN toan bo do dai noi dung cau tra loi."""
    if not raw_answer:
        return "Không rõ"
        
    ans = raw_answer.strip()
    ans = re.sub(r'^["\']+|["\']+$', '', ans).strip()
    
    prefix_patterns = [
        r'^(câu thơ|hai câu thơ|bài thơ|tiêu đề|tên món ăn|tên của xã|tên xã|đáp án|câu trả lời|kết quả)(\s+của\s+[^:]+)?\s*(là|đó là|chính là)?\s*[:\-\.]?\s*',
        r'^(dựa vào|nhìn vào|theo|quan sát)\s+hình ảnh\s*[,:]?\s*(thì|ta thấy|có thể thấy)?\s*',
        r'^(trong hình|trên hình|trong ảnh|trên ảnh)\s*[,:]?\s*',
        r'^(đó là|nó là|chính là|là)\s*[:\-\.]?\s*',
        r'^đáp án\s*(là)?\s*[:\-\.]?\s*',
        r'^trả lời\s*(là)?\s*[:\-\.]?\s*'
    ]
    
    changed = True
    while changed:
        changed = False
        for pat in prefix_patterns:
            new_ans = re.sub(pat, '', ans, flags=re.IGNORECASE).strip()
            if new_ans != ans and new_ans:
                ans = new_ans
                changed = True
                
    ans = ans.strip().strip('"').strip("'").rstrip('.!?;:')
    return ans if ans else "Không rõ"

def solve_task2(query_text, question, fused_candidates, keyframes_dir, model_id="Qwen/Qwen2-VL-2B-Instruct", metadata_dir=None, object_searcher=None):
    """
    Giai quyet Task 2 (Visual Q&A) bang co che Multi-Frame Visual Reasoning:
    1. Nap chuoi 3-4 keyframes tieu bieu (Intro/Signboard, Peak, Context) vao Qwen2-VL.
    2. VLM quan sat toan chuoi anh de suy luan khach quan theo bang chung thi giac thuc te.
    """
    if not fused_candidates:
        return {"video_id": "none", "frame_id": "0000", "answer": "không rõ"}
        
    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    dense_info = best_candidate.get("dense_info")
    
    scores = dense_info.get("all_scores") if dense_info is not None else None
    
    # 1. Chon chuoi 3-4 chi so frame tieu bieu (Multi-Frame Candidates)
    frame_indices = []
    if scores is not None and len(scores) > 0:
        n_frames = len(scores)
        peak_idx = int(np.argmax(scores))
        frame_indices.append(peak_idx)
        
        # Frame dau video (thuong chua bien hieu, tieu de, cong chao)
        intro_idx = max(0, min(5, n_frames // 10))
        if intro_idx not in frame_indices:
            frame_indices.append(intro_idx)
            
        # Frame dinh phu cach xa dinh chinh
        masked_scores = np.copy(scores)
        start_mask = max(0, peak_idx - 12)
        end_mask = min(n_frames, peak_idx + 12)
        masked_scores[start_mask:end_mask] = -np.inf
        if np.max(masked_scores) > -np.inf:
            sec_peak = int(np.argmax(masked_scores))
            if sec_peak not in frame_indices:
                frame_indices.append(sec_peak)
                
        # Frame giua video
        mid_idx = n_frames // 2
        if mid_idx not in frame_indices:
            frame_indices.append(mid_idx)
    else:
        frame_indices = [0]
        
    frame_indices.sort()
    best_frame_idx = frame_indices[0] if frame_indices else 0
    frame_id = get_frame_id_from_idx(keyframes_dir, video_id, best_frame_idx, metadata_dir=metadata_dir)
    
    # 2. Dinh vi cac file anh vat ly
    selected_image_paths = []
    for f_idx in frame_indices:
        p = find_image_for_frame(keyframes_dir, video_id, f_idx, frame_id)
        if p and p not in selected_image_paths:
            selected_image_paths.append(p)
            
    if not selected_image_paths:
        print(f"VQA Canh bao: Khong the dinh vi anh cho video {video_id}")
        return {"video_id": video_id, "frame_id": frame_id, "answer": "không rõ"}
        
    print(f"VQA: Suy luan Multi-Frame cho video {video_id} tren {len(selected_image_paths)} khung hinh...")
    model, processor = load_vlm(model_id)

    # 3. Ve sinh cau hoi thi giac
    clean_question = question
    for phrase in ["trong đoạn video có", "trong đoạn video", "đoạn video về", "đoạn video có", "đoạn video", "trong video", "video", "clip"]:
        clean_question = re.sub(r'\b' + re.escape(phrase) + r'\b', 'hình ảnh', clean_question, flags=re.IGNORECASE)
        
    prompt_text = (
        f"Quan sát các hình ảnh từ video trên. Hãy đọc các dòng chữ, biển hiệu, phông nền hoặc chi tiết thị giác để trả lời câu hỏi sau bằng Tiếng Việt:\n"
        f"'{clean_question}'\n"
        f"Trả lời trực tiếp vào trọng tâm câu hỏi, không giải thích."
    )
    
    # 4. Tao cau truc message da anh (Multi-Image Input)
    content_list = []
    for img_p in selected_image_paths:
        content_list.append({"type": "image", "image": img_p})
    content_list.append({"type": "text", "text": prompt_text})
    
    messages = [{"role": "user", "content": content_list}]
    
    raw_answer = ""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
            generated_ids = model.generate(**inputs, max_new_tokens=100)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            raw_answer = output_text[0].strip()
    except Exception as e:
        print(f"VQA Canh bao suy luan ({e}).")
        raw_answer = "Không rõ"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"VQA Dap an: '{raw_answer}'")
    final_answer = clean_vlm_answer(raw_answer)
    
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "answer": final_answer
    }
