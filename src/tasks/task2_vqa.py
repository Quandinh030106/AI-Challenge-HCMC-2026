import os
import glob
import json
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
            
        # Gioi han pixel de tranh bùng no tokens gay OOM tren GPU T4
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        try:
            _vlm_processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        except Exception:
            _vlm_processor = AutoProcessor.from_pretrained(model_id)
        print("VQA: Khoi tao mo hinh VLM thanh cong (toi uu VRAM).")
    return _vlm_model, _vlm_processor

def solve_task2(query_text, question, fused_candidates, keyframes_dir, model_id="Qwen/Qwen2-VL-2B-Instruct", metadata_dir=None, object_searcher=None):


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
        
    frame_id = get_frame_id_from_idx(keyframes_dir, video_id, best_frame_idx, metadata_dir=metadata_dir)
    
    # Tim file anh vat ly tren dia
    level = video_id.split('_')[0] if '_' in video_id else ""
    idx_4d = f"{best_frame_idx:04d}"
    idx_5d = f"{best_frame_idx:05d}"
    idx_raw = str(best_frame_idx)
    idx_1based = f"{best_frame_idx + 1:04d}"
    fid_str = f"{int(frame_id):04d}" if str(frame_id).isdigit() else str(frame_id)
    
    candidate_img_names = [idx_4d, idx_5d, idx_raw, idx_1based, fid_str]
    candidate_img_paths = []
    for name in candidate_img_names:
        candidate_img_paths.extend([
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, level, "keyframes", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, "keyframes", video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, video_id, f"{name}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{name}.jpeg"),
            os.path.join(keyframes_dir, video_id, f"{name}.png")
        ])
    
    image_path = None
    for p in candidate_img_paths:
        if os.path.exists(p):
            image_path = p
            break
            
    if not image_path:
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
                    target_idx = min(best_frame_idx, len(all_imgs) - 1)
                    image_path = all_imgs[target_idx]
                    break
                    
    if not image_path:
        print(f"VQA Canh bao: Khong the dinh vi anh cho video {video_id}")
        return {"video_id": video_id, "frame_id": frame_id, "answer": "không rõ"}
        
    # Doc them OCR/Metadata cua video neu co
    ocr_context = ""
    if metadata_dir and os.path.exists(metadata_dir):
        json_candidates = [
            os.path.join(metadata_dir, f"{video_id}.json"),
            os.path.join(metadata_dir, f"{video_id}_ocr.json"),
            os.path.join(metadata_dir, "media-info", f"{video_id}.json"),
            os.path.join(metadata_dir, "media-info-aic25-b1", "media-info", f"{video_id}.json")
        ]
        for jc in json_candidates:
            if os.path.exists(jc):
                try:
                    with open(jc, "r", encoding="utf-8") as jf:
                        jdata = json.load(jf)
                        ocr_context = " ".join([str(v) for v in jdata.values() if isinstance(v, (str, list))])[:300]
                        break
                except Exception:
                    pass

    print(f"VQA: Suy luan cau hoi cho video {video_id} tren anh {os.path.basename(image_path)}...")
    model, processor = load_vlm(model_id)

    hint_text = ""
    if object_searcher:
        objs = object_searcher.get_frame_objects(video_id, best_frame_idx)
        if objs:
            top_entities = [o["entity"] for o in objs[:8]]
            hint_text = f" (Vật thể: {', '.join(top_entities)})."
    
    ocr_hint = f" (Chữ OCR nhận diện trong video: {ocr_context})" if ocr_context else ""
    
    # Ve sinh cau hoi: Thay the 'doan video' thanh 'hinh anh' de Qwen2-VL khong bao gio bi trigger cau tu choi
    clean_question = question
    for phrase in ["trong đoạn video có", "trong đoạn video", "đoạn video về", "đoạn video có", "đoạn video", "trong video", "video", "clip"]:
        clean_question = re.sub(r'\b' + re.escape(phrase) + r'\b', 'hình ảnh', clean_question, flags=re.IGNORECASE)
        
    prompt_text = (
        f"Quan sát thật kỹ bức ảnh này{hint_text}{ocr_hint}. Hãy đọc các chữ trên biển hiệu, phông nền, tiêu đề hoặc hình ảnh để trả lời câu hỏi sau bằng Tiếng Việt:\n"
        f"'{clean_question}'\n"
        f"Trả lời ngắn gọn, trực tiếp vào trọng tâm, không giải thích."
    )


    
    raw_answer = ""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
        print(f"VQA Canh bao suy luan ({e}), chuyen sang che do fallback OCR/Metadata.")
        raw_answer = ocr_context if ocr_context else "Không rõ"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    print(f"VQA Dap an: '{raw_answer}'")

    clean_ans = raw_answer
    for prefix in ["đáp án:", "đáp án là:", "trả lời:", "câu trả lời:", "là", "đó là", "nó là"]:
        if clean_ans.lower().startswith(prefix):
            clean_ans = clean_ans[len(prefix):].strip()
            
    clean_ans = clean_ans.rstrip('.!?,;:')
    if not clean_ans or any(k in clean_ans.lower() for k in ["xin lỗi", "không thể xác định", "không thể cung cấp"]):
        clean_ans = ocr_context[:100] if ocr_context else "Không rõ"
        
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "answer": clean_ans
    }


