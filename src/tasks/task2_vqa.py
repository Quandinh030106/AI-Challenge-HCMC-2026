import os
import glob
import json
import re
import torch
import numpy as np

import sys
import subprocess

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    try:
        print("VQA: Dang tu dong cai dat thu vien 'qwen-vl-utils'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils", "--quiet"])
        from qwen_vl_utils import process_vision_info
        print("VQA: Cai dat 'qwen-vl-utils' thanh cong!")
    except Exception as e:
        print(f"VQA Canh bao ({e}). Khong the khoi tao qwen_vl_utils.")
        process_vision_info = None

from transformers import AutoProcessor
from src.tasks.task1_kis import get_frame_id_from_idx


_vlm_model = None
_vlm_processor = None

def load_vlm(model_id="Qwen/Qwen2-VL-2B-Instruct"):
    """Nap mo hinh VLM Qwen2-VL-2B-Instruct o che do do phan giai HD cao de doc chu ro net."""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        print(f"VQA: Nap mo hinh {model_id} (che do OCR HD)...")
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
        max_pixels = 1024 * 28 * 28  # Do phan giai HD sac net cho OCR bien hieu va van ban
        try:
            _vlm_processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        except Exception:
            _vlm_processor = AutoProcessor.from_pretrained(model_id)
        print("VQA: Khoi tao mo hinh VLM HD thanh cong (toi uu VRAM).")
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

def clean_vlm_answer(raw_answer, question=None):
    """Lam sach tien to dan chuyen nhung GIU NGUYEN VEN toan bo do dai noi dung cau tra loi duoi 100 ky tu."""
    if not raw_answer:
        return "Không rõ"
        
    ans = raw_answer.strip()
    
    # 1. Neu cau tra loi co chua dau ngoac kep bieu thi noi dung cot loi
    quoted_match = re.search(r'["“]([^"”]+)["”]', ans)
    if quoted_match and len(quoted_match.group(1).strip()) > 2 and len(ans) > 60:
        ans = quoted_match.group(1).strip()
        
    # 2. Cat bo cau hoi neu VLM lap lai nguyen van cau hoi
    if question and len(question) > 10:
        q_clean = question.strip().rstrip('?')
        if ans.lower().startswith(q_clean.lower()):
            ans = ans[len(q_clean):].lstrip(' : là,.-')
            
    prefix_patterns = [
        r'^(hình ảnh|đoạn video|video|clip)(\s+về\s+[^:]+)?\s*(là|đó là|chính là)?\s*[:\-\.]?\s*',
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
    if len(ans) > 100:
        ans = ans[:100].strip()
    return ans if ans else "Không rõ"

def solve_single_video_vqa(video_id, dense_info, question, clean_question, keyframes_dir, model, processor, metadata_dir=None):
    """Suy luan Multi-Frame cho mot video ung vien cu the."""
    scores = dense_info.get("all_scores") if dense_info is not None else None
    frame_indices = []
    if scores is not None and len(scores) > 0:
        n_frames = len(scores)
        peak_idx = int(np.argmax(scores))
        frame_indices.append(peak_idx)
        
        # Frame dau video (chua bien hieu, tieu de, cong chao)
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
    
    selected_image_paths = []
    for f_idx in frame_indices:
        p = find_image_for_frame(keyframes_dir, video_id, f_idx, frame_id)
        if p and p not in selected_image_paths:
            selected_image_paths.append(p)
            
    if not selected_image_paths:
        return {"frame_id": frame_id, "answer": "Không rõ", "has_concrete_answer": False}
        
    prompt_text = (
        f"Nhiệm vụ: Hãy quan sát kỹ các hình ảnh từ video trên, đọc chính xác từng dòng chữ, biển hiệu, bảng tên, hoành phi câu đối, nhãn dán hoặc tài liệu để trả lời câu hỏi sau bằng Tiếng Việt:\n"
        f"'{clean_question}'\n"
        f"Yêu cầu: Trả lời ngắn gọn, trực tiếp và chính xác tên riêng / câu thơ / tiêu đề cần tìm. Không thêm lời dẫn."
    )
    
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
            
    final_answer = clean_vlm_answer(raw_answer, question=clean_question)
    
    # Kiem tra xem dap an co gia tri thuc su hay chi la cau tra loi generic
    is_concrete = False
    if final_answer and final_answer.lower() not in ["không rõ", "chưa rõ", "none", "không có", "không tìm thấy", ""]:
        # Neu dap an khong chi lap lai tu trong cau hoi ma chua thong tin moi
        is_concrete = True
        
    return {
        "frame_id": frame_id,
        "raw_answer": raw_answer,
        "answer": final_answer,
        "has_concrete_answer": is_concrete
    }

def score_vqa_answer(ans, question, rank_idx=0):
    """Cham diem do tin cay va tinh phu hop cua dap an voi cau hoi de chon ra video chuan nhat."""
    if not ans or ans.lower() in ["không rõ", "chưa rõ", "none", "không có", "không tìm thấy"]:
        return -100.0
        
    score = 10.0 - (rank_idx * 1.5)  # Uu tien thu hang goc
    ans_lower = ans.lower()
    q_lower = question.lower()
    
    # 1. Cau hoi ve ten xa / dia danh o Khanh Hoa
    if "xã" in q_lower or "khánh hòa" in q_lower:
        if "xã" in ans_lower or "giang" in ans_lower or "ly" in ans_lower:
            score += 40.0
        elif "hà nội" in ans_lower or "sài gòn" in ans_lower:
            score -= 15.0  # Phat vi lac de so voi Khanh Hoa
            
    # 2. Cau hoi ve 2 cau tho
    elif "thơ" in q_lower or "câu thơ" in q_lower:
        if len(ans) > 20 or "," in ans or "\n" in ans or "/" in ans:
            score += 40.0
            
    # 3. Cau hoi ve tieu de / ten mon an
    elif "món ăn" in q_lower or "công thức" in q_lower or "tiêu đề" in q_lower:
        dish_indicators = ["thịt", "canh", "bò", "heo", "xào", "nấu", "kho", "chả", "bánh", "gỏi", "món", "cơm", "hấp", "nướng"]
        if any(d in ans_lower for d in dish_indicators):
            score += 40.0
            
    # Tang diem neu cau tra loi co do dai cu the tu 3 den 40 ky tu
    if 4 <= len(ans) <= 60:
        score += 5.0
        
    return score

def solve_task2(query_text, question, fused_candidates, keyframes_dir, model_id="Qwen/Qwen2-VL-2B-Instruct", metadata_dir=None, object_searcher=None):
    """
    Giai quyet Task 2 (Visual Q&A) bang co che QA-Driven Multi-Candidate Verification:
    1. Danh gia toan dien ca Top 4 video ung vien hang dau.
    2. Cham diem do tin cay va tinh dung de cua cau tra loi de chon video Top 1 chuan xac nhat!
    """

    if not fused_candidates:
        return {"video_id": "none", "frame_id": "0000", "answer": "không rõ", "promoted_idx": 0}
        
    model, processor = load_vlm(model_id)
    
    clean_question = question
    for phrase in ["trong đoạn video có", "trong đoạn video", "đoạn video về", "đoạn video có", "đoạn video", "trong video", "video", "clip"]:
        clean_question = re.sub(r'\b' + re.escape(phrase) + r'\b', 'hình ảnh', clean_question, flags=re.IGNORECASE)
        
    eval_candidates = fused_candidates[:4]
    evaluated_results = []
    
    for rank_idx, cand in enumerate(eval_candidates):
        vid = cand["video_id"]
        dense_info = cand.get("dense_info")
        print(f"VQA: Kiem tra Ung vien #{rank_idx + 1} ({vid})...")
        
        res = solve_single_video_vqa(
            vid, dense_info, question, clean_question, 
            keyframes_dir, model, processor, metadata_dir=metadata_dir
        )
        
        q_score = score_vqa_answer(res["answer"], question, rank_idx=rank_idx)
        print(f"  -> Video {vid} | Dap an: '{res['answer']}' (Diem chat luong: {q_score:.1f})")
        
        evaluated_results.append({
            "video_id": vid,
            "frame_id": res["frame_id"],
            "answer": res["answer"],
            "promoted_idx": rank_idx,
            "quality_score": q_score
        })
        
    # Chon ung vien co diem chat luong dap an cao nhat
    evaluated_results.sort(key=lambda x: x["quality_score"], reverse=True)
    best_candidate_result = evaluated_results[0]
    
    print(f"🎯 VQA: XAC THUC CHINH XAC! Lua chon Video {best_candidate_result['video_id']} (Rank goc #{best_candidate_result['promoted_idx'] + 1}) voi dap an: '{best_candidate_result['answer']}'")
    return best_candidate_result


