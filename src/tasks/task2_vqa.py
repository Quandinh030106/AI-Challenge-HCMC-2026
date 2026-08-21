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
    """Nap duy nhat 1 instance mo hinh VLM chia se toan bo he thong, tu dong gan len GPU 1 neu co Dual-GPU."""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        print(f"VLM: Nap mo hinh {model_id} (Shared Singleton Instance)...")
        
        # Neu co tu 2 GPU tro len tren Kaggle, uu tien dat VLM sang cuda:1 de tach biet voi CLIP o cuda:0
        if torch.cuda.is_available():
            if torch.cuda.device_count() >= 2:
                device_map = {"": "cuda:1"}
                print("VLM: Phat hien Dual-GPU -> Tu dong gan VLM doc quyen tren cuda:1 (15GB VRAM)!")
            else:
                device_map = "auto"
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            torch_dtype = torch.float32
        
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
        print("VLM: Khoi tao mo hinh VLM thanh cong (Khong nhan ban, toi uu VRAM tuyet doi).")
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
    """Lam sach tien to dan chuyen va loai bo chatbot refusals / placeholders."""
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
        r'^(?:hình ảnh|đoạn video|video|clip)(?:\s+[^:–\-]+?)?\s*(?:là|đó là|chính là)?\s*[:\-\.]?\s*',
        r'^(?:câu thơ|hai câu thơ|bài thơ|tiêu đề|tên(?:\s+của)?\s+[^:–\-]+?|đáp án|câu trả lời|kết quả)\s*(?:là|đó là|chính là)\s*[:\-\.]?\s*',
        r'^(?:câu thơ|hai câu thơ|bài thơ|tiêu đề|tên(?:\s+của)?\s+[^:–\-]+?|đáp án|câu trả lời|kết quả)\s*[:\-\.]\s*',
        r'^(?:món ăn|món|công thức|tiêu đề)(?:\s+này)?(?:\s+có)?(?:\s+tiêu đề|\s+tên)?\s*(?:là|đó là|chính là)\s*[:\-\.]?\s*',
        r'^(?:xã|huyện|tỉnh|địa phương|nơi)(?:\s+này)?(?:\s+có)?(?:\s+tên)?\s*(?:là|đó là|chính là)\s*[:\-\.]?\s*',
        r'^(?:dựa vào|nhìn vào|theo|quan sát)\s+hình ảnh\s*[,:]?\s*(?:thì|ta thấy|có thể thấy)?\s*',
        r'^(?:trong hình|trên hình|trong ảnh|trên ảnh)\s*[,:]?\s*',
        r'^(?:đó là|nó là|chính là|là)\s*[:\-\.]?\s*',
        r'^đáp án\s*(?:là)?\s*[:\-\.]?\s*',
        r'^trả lời\s*(?:là)?\s*[:\-\.]?\s*'
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
    
    # 3. Bo loc tu choi Chatbot va Placeholder pho bien
    refusal_keywords = [
        "không rõ", "chưa rõ", "none", "không có", "không tìm thấy", "không thể",
        "tôi không", "xin lỗi", "hãy xin lỗi", "tôi không hiểu", "không hiểu",
        "không thấy", "không xác định", "xã a", "xã b", "món a", "món b", "người a", "người b", "công thức a"
    ]
    ans_lower = ans.lower()
    if any(k in ans_lower for k in refusal_keywords):
        return "Không rõ"
        
    if len(ans) > 100:
        ans = ans[:100].strip()
    return ans if ans else "Không rõ"

def select_comprehensive_keyframes(dense_info, total_budget=8):
    """
    Trich xuat tap hop 6-8 khung hinh bao quat toan dien ca ve chieu sau (Peaks) va chieu rong (Temporal Coverage):
    1. Cac moc thoi gian trai deu toan video (Bao quat chieu rong toan bo dien bien 5%, 25%, 50%, 75%, 95%).
    2. Cac dinh cuc dai trong tung phan khuc Dau - Giua - Cuoi video (Chieu sau ngu nghia).
    3. Dinh cuc dai toan cuc va vung lan can (+-1..2 frames) de bat goc quay can canh.
    """
    if not dense_info or "all_scores" not in dense_info:
        return [0]
        
    scores = dense_info["all_scores"]
    n_frames = len(scores)
    if n_frames <= total_budget:
        return list(range(n_frames))
        
    from src.tasks.task1_kis import gaussian_smooth_scores
    smoothed = gaussian_smooth_scores(scores, sigma=1.5)
    
    selected_indices = set()
    
    # 1. Cac moc thoi gian trai deu toan video
    for ratio in [0.05, 0.25, 0.50, 0.75, 0.95]:
        f = int(n_frames * ratio)
        if 0 <= f < n_frames:
            selected_indices.add(f)
            
    # 2. Cac dinh cuc dai trong 3 phan khuc Dau - Giua - Cuoi
    segment_len = n_frames // 3
    for seg_i in range(3):
        start_s = seg_i * segment_len
        end_s = (seg_i + 1) * segment_len if seg_i < 2 else n_frames
        if start_s < end_s:
            local_peak = start_s + int(np.argmax(smoothed[start_s:end_s]))
            selected_indices.add(local_peak)
            
    # 3. Dinh cuc dai toan cuc va vung lan can (+-1, +-2 frames)
    global_peak = int(np.argmax(smoothed))
    selected_indices.add(global_peak)
    for delta in [-2, -1, 1, 2]:
        p = global_peak + delta
        if 0 <= p < n_frames:
            selected_indices.add(p)
            
    sorted_indices = sorted(list(selected_indices))
    if len(sorted_indices) > total_budget:
        # Uu tien giu lai cac frame co diem tuong dong cao nhat kem moc trai deu
        peak_priority = sorted(sorted_indices, key=lambda idx: smoothed[idx], reverse=True)
        sorted_indices = sorted(peak_priority[:total_budget])
        
    return sorted_indices

def solve_single_video_vqa(video_id, dense_info, query_text, question, clean_question, keyframes_dir, model, processor, metadata_dir=None):
    """Suy luan Multi-Frame can bang toan dien bao quat ca chieu rong va chieu sau theo dong thoi gian."""
    frame_indices = select_comprehensive_keyframes(dense_info, total_budget=8)
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
        f"Bối cảnh video: {query_text}\n"
        f"Câu hỏi: {clean_question}\n"
        f"Yêu cầu: Dựa vào các hình ảnh trên, hãy quan sát kỹ các chi tiết, chữ viết, biển hiệu hoặc hành động để trả lời câu hỏi bằng Tiếng Việt.\n"
        f"- Trả lời ngắn gọn, trực tiếp và chính xác đáp án cần tìm (dưới 20 từ). Không giải thích dài dòng.\n"
        f"- Nếu trong hình ảnh KHÔNG có thông tin hoặc KHÔNG nhìn thấy câu trả lời, hãy chỉ trả lời duy nhất: 'Không rõ'."
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
            generated_ids = model.generate(**inputs, max_new_tokens=80)
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
    
    is_concrete = False
    if final_answer and final_answer.lower() not in ["không rõ", "chưa rõ", "none", "không có", "không tìm thấy", ""]:
        is_concrete = True
        
    return {
        "frame_id": frame_id,
        "raw_answer": raw_answer,
        "answer": final_answer,
        "has_concrete_answer": is_concrete
    }

def score_vqa_answer(ans, question, rank_idx=0):
    """
    Cham diem do tin cay cua dap an theo nguyen ly tong quat 100%:
    - Dap an khong co thong tin / generic / refusal -> -100 diem.
    - Dap an ro rang, co noi dung cu the -> Uu tien cao nhat theo thu hang goc RRF.
    """
    if not ans or ans.lower() in ["không rõ", "chưa rõ", "none", "không có", "không tìm thấy", ""]:
        return -100.0
        
    # Diem co ban dua theo thu hang tim kiem goc RRF
    score = 50.0 - (rank_idx * 3.0)
    
    # Cong diem neu dap an co do dai chuan gon (tu 3 den 70 ky tu)
    if 3 <= len(ans) <= 70:
        score += 10.0
    elif len(ans) > 100:
        score -= 10.0
        
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
            vid, dense_info, query_text, question, clean_question, 
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


