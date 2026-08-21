import os
import glob
import re
import numpy as np
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

class VisualReRanker:
    """Module xac thuc thi giac su dung Qwen-VL de cham diem lai Top ung vien."""
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        
    def _load_model(self):
        if self.model is None or self.processor is None:
            from src.tasks.task2_vqa import load_vlm
            self.model, self.processor = load_vlm(self.model_id)





    def find_keyframe_image(self, keyframes_dir, video_id, frame_idx):
        """Tim file anh vat ly cua video tren dia."""
        level = video_id.split('_')[0] if '_' in video_id else ""
        idx_4d = f"{frame_idx:04d}"
        idx_5d = f"{frame_idx:05d}"
        idx_raw = str(frame_idx)
        idx_1based = f"{frame_idx + 1:04d}"
        
        candidate_img_paths = [
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_4d}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", video_id, f"{idx_4d}.jpg"),
            os.path.join(keyframes_dir, level, "keyframes", video_id, f"{idx_4d}.jpg"),
            os.path.join(keyframes_dir, "keyframes", video_id, f"{idx_4d}.jpg"),
            os.path.join(keyframes_dir, video_id, f"{idx_4d}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_raw}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_1based}.jpg"),
            os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_5d}.jpg")
        ]
        
        for p in candidate_img_paths:
            if os.path.exists(p):
                return p
                
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

    def verify_single_image(self, image_path, query_text):
        """Đánh giá độ trùng khớp của ảnh trên thang điểm 0-10."""
        self._load_model()
        prompt = (
            f"Hãy quan sát kỹ bức ảnh này và cho biết: Bức ảnh có chứa đúng phân cảnh được mô tả sau đây không: "
            f"'{query_text}'?\n"
            f"Đánh giá độ trùng khớp trên thang điểm từ 0 đến 10 (10 = hoàn toàn trùng khớp chi tiết, 0 = hoàn toàn không liên quan).\n"
            f"Trả về kết quả theo định dạng: Score: X (ví dụ: Score: 9.5 hoặc Score: 2)."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=60,       # Tăng token để VLM kịp trả về chuỗi hoàn chỉnh
                do_sample=False,         # Tắt lấy mẫu ngẫu nhiên
                temperature=0.0,         # Cố định câu trả lời tuyệt đối
                top_p=1.0,
                repetition_penalty=1.1
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
        ans = output_text[0].strip()
        
        # Chỉ lấy số đứng sau định dạng "Score: X"
        score_match = re.search(r'Score:\s*([0-9]+(?:\.[0-9]+)?)', ans, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Nếu không tìm thấy tiền tố Score:, mới xét từ khóa
            score = 8.0 if any(w in ans.lower() for w in ["yes", "có", "đúng", "match"]) else 1.0
            
        return min(max(score, 0.0), 10.0)

    def rerank_candidates(self, fused_candidates, query_text, keyframes_dir, top_n_verify=5):
        """Xac thuc thi giac cho Top N video ung vien va sap xep lai thu hang."""
        if not fused_candidates:
            return []
            
        candidates_to_verify = fused_candidates[:top_n_verify]
        remaining_candidates = fused_candidates[top_n_verify:]
        
        verified_results = []
        for cand in candidates_to_verify:
            vid = cand["video_id"]
            dense_info = cand.get("dense_info")
            best_idx = dense_info.get("best_frame_idx", 0) if dense_info else 0
            
            img_path = self.find_keyframe_image(keyframes_dir, vid, best_idx)
            if img_path and os.path.exists(img_path):
                try:
                    vlm_score = self.verify_single_image(img_path, query_text)
                except Exception:
                    vlm_score = 5.0
            else:
                vlm_score = 0.0
                
            original_rrf = cand.get("rrf_score", 0.0)
            boosted_score = original_rrf + (vlm_score / 10.0) * 0.15
            
            cand_copy = dict(cand)
            cand_copy["vlm_score"] = vlm_score
            cand_copy["boosted_score"] = boosted_score
            verified_results.append(cand_copy)
            
        verified_results.sort(key=lambda x: x["boosted_score"], reverse=True)
        return verified_results + remaining_candidates
