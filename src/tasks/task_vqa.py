# ==============================================================================
# AIC 2026 - VISUAL QUESTION ANSWERING (VQA) SOLVER WITH TOP-1 PROMOTION
# ==============================================================================
import os
import re
import torch
from PIL import Image
from typing import List, Dict, Any

def clean_vqa_answer(raw_answer: str, question: str = "") -> str:
    """
    Cleans raw VLM answer strings using general linguistic patterns.
    Strips conversational prefixes, quotes, and limits output to 100 characters.
    """
    if not raw_answer or str(raw_answer).strip().lower() in ["none", "null", "chưa rõ", "không rõ", ""]:
        return "Không rõ"

    ans = str(raw_answer).strip().replace("\n", " ").strip()

    generic_declarative_pattern = r'^(dựa vào|theo|qua|quan sát|dựa trên|trong|trên)\s+[^\s,.:;]+\s*,?\s*'
    ans = re.sub(generic_declarative_pattern, '', ans, flags=re.IGNORECASE).strip()

    generic_result_pattern = r'^(kết quả|đáp án|câu trả lời|trả lời|thông tin|nội dung|giá trị|chi tiết)\s*[^\s,.:;]*\s*(là|được ghi|hiển thị|thấy được)?\s*[:\-\"]*\s*'
    ans = re.sub(generic_result_pattern, '', ans, flags=re.IGNORECASE).strip()

    refusal_pattern = r'^(không thể|chưa thể|không thấy|khó|không có|chưa có|không xác định)\s+.*$'
    if re.search(refusal_pattern, ans, re.IGNORECASE):
        return "Không rõ"

    ans = ans.strip().strip('"').strip("'").strip('`').rstrip('.!?;:')
    if len(ans) > 100:
        ans = ans[:100].strip()
    return ans if ans else "Không rõ"

def format_vqa_answer_for_csv(ans: str) -> str:
    """Formats answer string conforming to Codabench CSV escaping rules."""
    clean_ans = clean_vqa_answer(ans)
    if ',' in clean_ans or '"' in clean_ans or '\n' in clean_ans:
        clean_ans = clean_ans.replace('"', '""')
        return f'"{clean_ans}"'
    return f'"{clean_ans}"'


class VisualVQASolver:
    """
    Solves Visual Q&A queries using Focused 3-Frame Local Temporal Window Sampling.
    Reads numbers on scales, bridge signs, mountain pass milestones, and map counts.
    Promotes high-confidence answered video to Top 1 to maximize R@1 = 1.0 (Final Score = 1.00).
    """
    def __init__(self, vlm_model=None, vlm_processor=None, db_manager=None, device="cuda:1"):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.db_manager = db_manager
        self.device = device

    def solve(
        self,
        parsed_schema: dict,
        candidates: List[Dict[str, Any]],
        total_preds: int = 100,
        eval_candidates_count: int = 4
    ) -> List[Dict[str, Any]]:
        """Solves VQA query and produces Top 100 candidate predictions with formatted answer."""
        if not candidates:
            return [{"video_id": f"L21_V{i:03d}", "frame_id": 0, "answer": '"Không rõ"'} for i in range(1, total_preds + 1)]

        vlm_question = parsed_schema.get("vlm_question", parsed_schema.get("query_vi", ""))
        final_candidates = list(candidates)

        best_cand_idx = 0
        best_score = -999.0
        best_answer = "Không rõ"

        # Evaluate Top candidate videos
        eval_pool = final_candidates[:eval_candidates_count]
        for rank_idx, cand in enumerate(eval_pool):
            vid = cand["video_id"]
            best_f_idx = int(cand.get("best_frame_idx", 0))

            # Retrieve 3-frame local temporal window [f_idx - 1, f_idx, f_idx + 1]
            image_paths = []
            if self.db_manager is not None:
                window_indices = [max(0, best_f_idx - 1), best_f_idx, best_f_idx + 1]
                frame_records = self.db_manager.fetch_frames_by_indices(vid, window_indices)
                for fr in frame_records:
                    p = fr.get("image_path", "")
                    if p and os.path.exists(p) and p not in image_paths:
                        image_paths.append(p)

            if not image_paths:
                single_p = cand.get("image_path", "")
                if single_p and os.path.exists(single_p):
                    image_paths.append(single_p)

            if not image_paths or self.vlm_model is None or self.vlm_processor is None:
                continue

            raw_ans = self._infer_vlm_multi_frame(image_paths, vlm_question)
            cleaned_ans = clean_vqa_answer(raw_ans, question=vlm_question)

            # Confidence scoring
            conf = 10.0 - (rank_idx * 1.5)
            if cleaned_ans and cleaned_ans.lower() not in ["không rõ", "none", "chưa rõ"]:
                conf += 30.0
                if any(c.isdigit() for c in cleaned_ans):
                    conf += 20.0

            if conf > best_score:
                best_score = conf
                best_cand_idx = rank_idx
                best_answer = cleaned_ans

        # Top-1 Promotion: Promote highest confidence answered video to Rank 1
        if best_cand_idx > 0 and best_score >= 25.0:
            promoted = final_candidates.pop(best_cand_idx)
            final_candidates.insert(0, promoted)
            print(f"[INFO] VQASolver: Promoted candidate '{promoted['video_id']}' to Top 1 (Answer: '{best_answer}').")

        formatted_ans = format_vqa_answer_for_csv(best_answer)

        predictions = []
        for cand in final_candidates[:total_preds]:
            vid = cand["video_id"]
            fid = int(cand.get("best_frame_id", cand.get("frame_id", 0)))
            predictions.append({
                "video_id": vid,
                "frame_id": fid,
                "answer": formatted_ans
            })

        return predictions

    def _infer_vlm_multi_frame(self, image_paths: List[str], question_text: str) -> str:
        """Runs multi-frame visual reasoning using Qwen2.5-VL with memory-safe downscaled frames."""
        prompt_text = (
            f"Nhiệm vụ: Quan sát kỹ các khung ảnh trải dài theo thời gian của video này và trả lời câu hỏi sau bằng Tiếng Việt:\n"
            f"'{question_text}'\n"
            f"Yêu cầu: Trả lời ngắn gọn, trực tiếp con số / tên riêng / từ cần tìm. Không thêm lời dẫn rườm rà."
        )

        content_items = []
        pil_images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((896, 896))
                pil_images.append(img)
                content_items.append({"type": "image", "image": img})
            except Exception:
                pass

        if not pil_images:
            return "Không rõ"

        content_items.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_items}]

        try:
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_device = next(self.vlm_model.parameters()).device if hasattr(self.vlm_model, "parameters") else self.device
            inputs = self.vlm_processor(text=[text], images=pil_images, padding=True, return_tensors="pt").to(model_device)

            with torch.inference_mode():
                gen_ids = self.vlm_model.generate(**inputs, max_new_tokens=50)
                gen_trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
                out_text = self.vlm_processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]

            del inputs, pil_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return out_text.strip()
        except Exception as e:
            print(f"[WARNING] VQA VLM Infer error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Không rõ"
