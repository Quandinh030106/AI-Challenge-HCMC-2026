import os
import glob
import json
import re
import numpy as np
import torch
from transformers import AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """
    Xác định Frame ID vật thể từ số chỉ số frame (0-based frame index).
    Hỗ trợ 0001.jpg, 00001.jpg, 1.jpg và file mapping CSV của BTC.
    """
    if not keyframes_dir or not os.path.exists(keyframes_dir):
        return f"{frame_idx:04d}"

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
            fname = os.path.splitext(os.path.basename(p))[0]
            return fname

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
                target_idx = min(max(0, frame_idx), len(all_imgs) - 1)
                fname = os.path.splitext(os.path.basename(all_imgs[target_idx]))[0]
                return fname

    return f"{frame_idx:04d}"


def clean_vlm_answer(raw_answer, question=""):
    """
    Làm sạch đáp án VLM
    Cắt sạch câu dẫn rườm rà, ép về đáp án cốt lõi ngắn gọn dưới 100 ký tự.
    """
    if not raw_answer or str(raw_answer).strip().lower() in ["none", "null", "chưa rõ", "không rõ", ""]:
        return "Không rõ"

    ans = str(raw_answer).strip().replace("\n", " ").strip()

    # Pattern 1: Xóa các tiền tố tuyên bố chung (Dựa vào..., Qua hình ảnh..., Đáp án là...)
    generic_declarative_pattern = r'^(dựa vào|theo|qua|quan sát|dựa trên|trong|trên)\s+[^\s,.:;]+\s*,?\s*'
    ans = re.sub(generic_declarative_pattern, '', ans, flags=re.IGNORECASE).strip()

    # Pattern 2: Xóa các cụm từ kết quả chung (Đáp án là, Kết quả là, Thông tin là...)
    generic_result_pattern = r'^(kết quả|đáp án|câu trả lời|trả lời|thông tin|nội dung|giá trị|chi tiết)\s*[^\s,.:;]*\s*(là|được ghi|hiển thị|thấy được)?\s*[:\-\"]*\s*'
    ans = re.sub(generic_result_pattern, '', ans, flags=re.IGNORECASE).strip()

    # Pattern 3: Xóa sự lặp lại câu hỏi tự động (Loại bỏ vế lặp từ câu hỏi nếu VLM nhắc lại)
    if question:
        clean_q_words = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', question) if len(w.strip()) >= 3]
        for q_w in clean_q_words[:3]:
            repeat_pat = r'^' + re.escape(q_w) + r'\s+.*?\s+(là|được ghi là|thể hiện là)\s*[:\-\"]*\s*'
            ans = re.sub(repeat_pat, '', ans, flags=re.IGNORECASE).strip()

    # Pattern 4: Xóa các thông điệp từ chối/không xác định chung
    refusal_pattern = r'^(không thể|chưa thể|không thấy|khó|không có|chưa có|không xác định)\s+.*$'
    if re.search(refusal_pattern, ans, re.IGNORECASE):
        return "Không rõ"

    # Chuẩn hóa trích xuất ký tự & làm sạch trích dẫn
    ans = ans.strip().strip('"').strip("'").strip('`').rstrip('.!?;:')
    if len(ans) > 100:
        ans = ans[:100].strip()
    return ans if ans else "Không rõ"


class TaskSolvers:
    """
    Bao gồm KIS Solver, VQA Heavy VLM Solver (Qwen2.5-VL-7B), và TRAKE Dynamic Programming Solver.
    """
    def __init__(self, keyframes_dir=None, metadata_dir=None, vlm_model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.keyframes_dir = keyframes_dir
        self.metadata_dir = metadata_dir
        self.vlm_model_id = vlm_model_id
        self.vlm_model = None
        self.vlm_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_vlm(self):
        if self.vlm_model is not None:
            return
            
        print(f"TaskSolvers: Đang nạp mô hình VLM ({self.vlm_model_id}) trên 2 GPU...")
        try:
            from transformers import Qwen2VLForConditionalGeneration
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.vlm_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        except Exception:
            from transformers import AutoModelForVision2Seq
            self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                self.vlm_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        try:
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_id, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
        except Exception:
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_id, trust_remote_code=True)

        self.vlm_model.eval()
        print("TaskSolvers: Nạp thành công Heavy VLM!")

    def unload_vlm(self):
        """Giải phóng VLM khỏi VRAM."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.vlm_processor is not None:
            del self.vlm_processor
            self.vlm_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("TaskSolvers: Đã giải phóng Heavy VLM khỏi VRAM!")

    # -------------------------------------------------------------------------
    # TASK 1: TEXTUAL KIS SOLVER
    # -------------------------------------------------------------------------
    def solve_kis(self, fused_candidates, total_preds=100):
        """Giải bài toán Textual KIS: Xuất danh sách 100 dự đoán đa dạng không dính frame 0."""
        predictions = []
        for rank, cand in enumerate(fused_candidates[:total_preds]):
            vid = cand["video_id"]
            dense_info = cand.get("dense_info", {})
            f_idx = dense_info.get("best_frame_idx", 0)
            
            fid = get_frame_id_from_idx(self.keyframes_dir, vid, f_idx, metadata_dir=self.metadata_dir)
            if not fid or fid in ["0", "0000", ""]:
                fid = f"{rank * 10:04d}"

            predictions.append({
                "video_id": vid,
                "frame_id": fid,
                "score": cand.get("rrf_score", 0.0)
            })
            
        return predictions

    # -------------------------------------------------------------------------
    # TASK 2: VISUAL Q&A SOLVER (Qwen2.5-VL-7B Heavy VLM)
    # -------------------------------------------------------------------------
    def solve_vqa(self, parsed_schema, fused_candidates):
        """
        Giải bài toán Visual Q&A bằng Heavy VLM Qwen2.5-VL-7B.
        Nhận vlm_question từ File 1, soi Top keyframes ứng viên và làm sạch đáp án.
        """
        if not fused_candidates:
            return {"video_id": "none", "frame_id": "0000", "answer": "Không rõ", "promoted_idx": 0}

        self.load_vlm()
        vlm_question = parsed_schema.get("vlm_question", parsed_schema.get("query_vi", ""))
        
        eval_candidates = fused_candidates[:4]
        best_candidate_idx = 0
        best_score = -999.0
        best_answer = "Không rõ"
        best_frame_id = "0000"

        for rank_idx, cand in enumerate(eval_candidates):
            vid = cand["video_id"]
            dense_info = cand.get("dense_info", {})
            f_idx = dense_info.get("best_frame_idx", 0)
            fid = get_frame_id_from_idx(self.keyframes_dir, vid, f_idx, metadata_dir=self.metadata_dir)

            img_path = self._find_keyframe_image(vid, f_idx, fid)
            if not img_path:
                continue

            raw_ans = self._infer_vlm_single_image(img_path, vlm_question)
            cleaned_ans = clean_vlm_answer(raw_ans, question=vlm_question)
            
            confidence_score = 10.0 - (rank_idx * 1.5)
            if cleaned_ans and cleaned_ans.lower() not in ["không rõ", "none", "chưa rõ"]:
                confidence_score += 30.0
                if any(c.isdigit() for c in cleaned_ans):
                    confidence_score += 20.0

            if confidence_score > best_score:
                best_score = confidence_score
                best_candidate_idx = rank_idx
                best_answer = cleaned_ans
                best_frame_id = fid

        return {
            "promoted_idx": best_candidate_idx,
            "answer": best_answer,
            "best_frame_id": best_frame_id
        }

    def _infer_vlm_single_image(self, image_path, question_text):
        """Suy luận trực tiếp bằng Qwen2.5-VL-7B trên 1 ảnh keyframe."""
        prompt_text = (
            f"Nhiệm vụ: Quan sát kỹ bức ảnh này và trả lời câu hỏi sau bằng Tiếng Việt:\n"
            f"'{question_text}'\n"
            f"Yêu cầu: Trả lời ngắn gọn, trực tiếp con số / tên riêng / từ cần tìm. Không thêm lời dẫn rườm rà."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text}
            ]
        }]
        try:
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = None, None

            inputs = self.vlm_processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(self.vlm_model.device)

            with torch.no_grad():
                gen_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)
                gen_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
                out_text = self.vlm_processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
                return out_text.strip()
        except Exception as e:
            print(f"VLM Infer Warning: {e}")
            return "Không rõ"

    def _find_keyframe_image(self, video_id, frame_idx, frame_id=""):
        """Tìm file ảnh vật lý trên ổ đĩa."""
        if not self.keyframes_dir or not os.path.exists(self.keyframes_dir):
            return None

        level = video_id.split('_')[0] if '_' in video_id else ""
        idx_4d = f"{frame_idx:04d}"
        
        candidates = [
            os.path.join(self.keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_4d}.jpg"),
            os.path.join(self.keyframes_dir, "keyframes", video_id, f"{idx_4d}.jpg"),
            os.path.join(self.keyframes_dir, video_id, f"{idx_4d}.jpg"),
            os.path.join(self.keyframes_dir, video_id, f"{frame_id}.jpg")
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    # -------------------------------------------------------------------------
    # TASK 3: TRAKE EVENT ALIGNMENT SOLVER (Dynamic Programming / Viterbi Alignment)
    # -------------------------------------------------------------------------
    def solve_trake(self, parsed_schema, fused_candidates, dense_engine=None, total_preds=100):
        """
        Giải bài toán TRAKE bằng Quy hoạch động (Dynamic Programming Viterbi Alignment).
        Đảm bảo thứ tự thời gian t_1 < t_2 < ... < t_N tăng dần tuyệt đối và xuất ĐỦ N frame IDs per row.
        """
        events = parsed_schema.get("bm25_keywords", [])
        if not events:
            events = [parsed_schema.get("query_vi", "")]

        n_events = len(events)
        aligned_results = []

        for cand in fused_candidates[:total_preds]:
            vid = cand["video_id"]
            dense_info = cand.get("dense_info", {})
            scores = dense_info.get("all_scores")

            if scores is not None and len(scores) >= n_events and dense_engine is not None:
                aligned_f_idxs = self._align_events_dp(events, scores, dense_engine)
            else:
                n_total = len(scores) if scores is not None else 100
                aligned_f_idxs = [int(x) for x in np.linspace(0, max(0, n_total - 1), n_events)]

            frame_ids = [
                get_frame_id_from_idx(self.keyframes_dir, vid, f_idx, metadata_dir=self.metadata_dir)
                for f_idx in aligned_f_idxs
            ]

            aligned_results.append({
                "video_id": vid,
                "frame_ids": frame_ids
            })

        return aligned_results

    def _align_events_dp(self, events, frame_scores, dense_engine):
        """Quy hoạch động Viterbi Alignment tìm chuỗi frame t_1 < t_2 < ... < t_N tối ưu nhất."""
        n_frames = len(frame_scores)
        n_events = len(events)

        if n_frames < n_events:
            return [int(x) for x in np.linspace(0, max(0, n_frames - 1), n_events)]

        dp = np.full((n_events, n_frames), -np.inf, dtype=np.float32)
        parent = np.zeros((n_events, n_frames), dtype=np.int32)

        dp[0, :] = frame_scores

        for e in range(1, n_events):
            for t in range(e, n_frames):
                best_prev_t = int(np.argmax(dp[e - 1, :t]))
                dp[e, t] = dp[e - 1, best_prev_t] + frame_scores[t]
                parent[e, t] = best_prev_t

        best_end_t = int(np.argmax(dp[n_events - 1, :]))
        aligned_idxs = [0] * n_events
        aligned_idxs[n_events - 1] = best_end_t

        curr_t = best_end_t
        for e in range(n_events - 1, 0, -1):
            curr_t = parent[e, curr_t]
            aligned_idxs[e - 1] = curr_t

        return aligned_idxs
