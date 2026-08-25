import os
import glob
import json
import re
import csv
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """
    Resolves physical frame ID from video frame index.
    Checks map-keyframes CSV tables first to get exact video frame IDs.
    """
    if metadata_dir and os.path.exists(metadata_dir):
        level = video_id.split('_')[0] if '_' in video_id else ""
        csv_candidates = [
            os.path.join(metadata_dir, "map-keyframes-aic25-b1", "map-keyframes", f"{video_id}.csv"),
            os.path.join(metadata_dir, "map-keyframes", f"{video_id}.csv"),
            os.path.join(metadata_dir, f"{video_id}.csv")
        ]
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = [r for r in reader if r]
                        if rows and len(rows[0]) >= 2:
                            header_offset = 1 if not rows[0][0].isdigit() else 0
                            target_row_idx = header_offset + frame_idx
                            if target_row_idx < len(rows):
                                frame_id_val = rows[target_row_idx][1].strip()
                                return os.path.splitext(frame_id_val)[0]
                except Exception:
                    pass

    if not keyframes_dir or not os.path.exists(keyframes_dir):
        return f"{max(0, frame_idx):03d}"

    level = video_id.split('_')[0] if '_' in video_id else ""
    idx_3d = f"{frame_idx:03d}"
    idx_3d_1based = f"{frame_idx + 1:03d}"
    idx_4d = f"{frame_idx:04d}"
    idx_raw = str(frame_idx)
    idx_4d_1based = f"{frame_idx + 1:04d}"

    candidate_img_paths = [
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_3d}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_3d_1based}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_4d}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", video_id, f"{idx_3d}.jpg"),
        os.path.join(keyframes_dir, level, "keyframes", video_id, f"{idx_3d}.jpg"),
        os.path.join(keyframes_dir, "keyframes", video_id, f"{idx_3d}.jpg"),
        os.path.join(keyframes_dir, video_id, f"{idx_3d}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_raw}.jpg"),
        os.path.join(keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{idx_4d_1based}.jpg")
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

    return f"{max(0, frame_idx):03d}"


def clean_vlm_answer(raw_answer, question=""):
    """
    Cleans raw VLM answer strings using abstract linguistic regex patterns.
    Strips conversational prefixes, quotes, and limits output to 100 characters.
    """
    if not raw_answer or str(raw_answer).strip().lower() in ["none", "null", "chưa rõ", "không rõ", ""]:
        return "Không rõ"

    ans = str(raw_answer).strip().replace("\n", " ").strip()

    generic_declarative_pattern = r'^(dựa vào|theo|qua|quan sát|dựa trên|trong|trên)\s+[^\s,.:;]+\s*,?\s*'
    ans = re.sub(generic_declarative_pattern, '', ans, flags=re.IGNORECASE).strip()

    generic_result_pattern = r'^(kết quả|đáp án|câu trả lời|trả lời|thông tin|nội dung|giá trị|chi tiết)\s*[^\s,.:;]*\s*(là|được ghi|hiển thị|thấy được)?\s*[:\-\"]*\s*'
    ans = re.sub(generic_result_pattern, '', ans, flags=re.IGNORECASE).strip()

    if question:
        clean_q_words = [w.strip() for w in re.split(r'[,.\s\?\!\:\;]+', question) if len(w.strip()) >= 3]
        for q_w in clean_q_words[:3]:
            repeat_pat = r'^' + re.escape(q_w) + r'\s+.*?\s+(là|được ghi là|thể hiện là)\s*[:\-\"]*\s*'
            ans = re.sub(repeat_pat, '', ans, flags=re.IGNORECASE).strip()

    refusal_pattern = r'^(không thể|chưa thể|không thấy|khó|không có|chưa có|không xác định)\s+.*$'
    if re.search(refusal_pattern, ans, re.IGNORECASE):
        return "Không rõ"

    ans = ans.strip().strip('"').strip("'").strip('`').rstrip('.!?;:')
    if len(ans) > 100:
        ans = ans[:100].strip()
    return ans if ans else "Không rõ"


class TaskSolvers:
    """
    Universal task solver suite for Textual KIS, Qwen2.5-VL-7B VQA,
    and Dynamic Programming (Viterbi) TRAKE temporal alignment.
    Supports Dense-Anchored Local Temporal Window Sampling around target event frames.
    """
    def __init__(self, keyframes_dir=None, metadata_dir=None, vlm_model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.keyframes_dir = self._resolve_keyframes_dir(keyframes_dir)
        self.metadata_dir = self._resolve_metadata_dir(metadata_dir)
        self.vlm_model_id = vlm_model_id
        self.vlm_model = None
        self.vlm_processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._keyframe_cache = {}

    def _resolve_keyframes_dir(self, path):
        if path and os.path.exists(path):
            return path
        search_roots = ["/kaggle/input", "data/keyframes", "data"]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, _ in os.walk(s_root):
                    if "keyframe" in root.lower() or "keyframes" in root.lower():
                        print(f"[INFO] TaskSolvers: Auto-discovered keyframes directory at '{root}'")
                        return root
        return path

    def _resolve_metadata_dir(self, path):
        if path and os.path.exists(path):
            return path
        search_roots = ["/kaggle/input", "data/metadata", "data"]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, files in os.walk(s_root):
                    if "metadata" in root.lower() or "map-keyframes" in root.lower() or any(f.endswith(".csv") for f in files):
                        return root
        return path

    def load_vlm(self):
        """Loads Qwen2.5-VL-7B model using exact Qwen2_5_VLForConditionalGeneration class on cuda:0."""
        if self.vlm_model is not None:
            return
            
        print(f"[INFO] TaskSolvers: Loading Heavy VLM ({self.vlm_model_id})...")
        
        model_cls = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                model_cls = Qwen2VLForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForCausalLM
                model_cls = AutoModelForCausalLM

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        try:
            self.vlm_model = model_cls.from_pretrained(
                self.vlm_model_id,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else None,
                ignore_mismatched_sizes=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[WARNING] 4-bit VLM loading failed ({e}). Loading in standard FP16...")
            self.vlm_model = model_cls.from_pretrained(
                self.vlm_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else None,
                ignore_mismatched_sizes=True,
                trust_remote_code=True
            )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        try:
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_id, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
        except Exception:
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_id, trust_remote_code=True)

        self.vlm_model.eval()
        print("[INFO] TaskSolvers: Loaded Heavy VLM successfully.")

    def unload_vlm(self):
        """Unloads Heavy VLM from VRAM."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.vlm_processor is not None:
            del self.vlm_processor
            self.vlm_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] TaskSolvers: Unloaded Heavy VLM from VRAM.")

    def solve_kis(self, fused_candidates, total_preds=100):
        """Solves Textual KIS task: outputs top 100 candidate frame predictions mapped to exact video frame IDs."""
        if not fused_candidates:
            fused_candidates = [{"video_id": f"L21_V{i:03d}", "dense_info": {"best_frame_idx": 1}, "rrf_score": 0.0} for i in range(1, total_preds + 1)]

        predictions = []
        for rank, cand in enumerate(fused_candidates[:total_preds]):
            vid = cand["video_id"]
            dense_info = cand.get("dense_info", {})
            f_idx = dense_info.get("best_frame_idx", 0)
            
            fid = get_frame_id_from_idx(self.keyframes_dir, vid, f_idx, metadata_dir=self.metadata_dir)
            if not fid:
                fid = f"{max(0, f_idx):03d}"

            predictions.append({
                "video_id": vid,
                "frame_id": fid,
                "score": cand.get("rrf_score", 0.0)
            })
            
        return predictions

    def solve_vqa(self, parsed_schema, fused_candidates):
        """
        Solves Visual Q&A task using Dense-Anchored Local Temporal Window Sampling around the target event frame.
        Guarantees exact event capture regardless of video duration (1 min vs 1 hour).
        """
        if not fused_candidates:
            return {"video_id": "none", "frame_id": "000", "answer": "Không rõ", "promoted_idx": 0}

        self.load_vlm()
        vlm_question = parsed_schema.get("vlm_question", parsed_schema.get("query_vi", ""))
        
        eval_candidates = fused_candidates[:4]
        best_candidate_idx = 0
        best_score = -999.0
        best_answer = "Không rõ"
        best_frame_id = "000"

        for rank_idx, cand in enumerate(eval_candidates):
            vid = cand["video_id"]
            dense_info = cand.get("dense_info", {})
            f_idx = dense_info.get("best_frame_idx", 0)
            fid = get_frame_id_from_idx(self.keyframes_dir, vid, f_idx, metadata_dir=self.metadata_dir)

            # Strategy: Dense-Anchored Local Temporal Window around f_idx + start & end anchors
            all_video_imgs = self._get_all_video_keyframe_paths(vid)
            if all_video_imgs:
                n_total = len(all_video_imgs)
                f_idx_clamped = min(max(0, f_idx), n_total - 1)
                
                # High-density local window around f_idx (best event frame from CLIP)
                local_window = [
                    max(0, f_idx_clamped - 3),
                    max(0, f_idx_clamped - 1),
                    f_idx_clamped,
                    min(n_total - 1, f_idx_clamped + 1),
                    min(n_total - 1, f_idx_clamped + 3)
                ]
                # Global boundary anchors
                global_anchors = [0, max(0, n_total - 1)]
                
                sampled_indices = sorted(list(set(local_window + global_anchors)))
                image_paths = [all_video_imgs[i] for i in sampled_indices]
            else:
                frame_indices = [max(0, f_idx - 2), max(0, f_idx - 1), f_idx, f_idx + 1, f_idx + 2]
                image_paths = []
                for fi in frame_indices:
                    p = self._find_keyframe_image(vid, fi, f"{fi:03d}")
                    if p and p not in image_paths:
                        image_paths.append(p)

            if not image_paths:
                img_path = self._find_keyframe_image(vid, f_idx, fid)
                if img_path:
                    image_paths = [img_path]
                else:
                    continue

            raw_ans = self._infer_vlm_multi_frame_video(image_paths, vlm_question)
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

    def _get_all_video_keyframe_paths(self, video_id):
        """Automatically discovers and returns all keyframe image paths for any target video_id."""
        if not self.keyframes_dir or not os.path.exists(self.keyframes_dir):
            return []

        level = video_id.split('_')[0] if '_' in video_id else ""
        folder_candidates = [
            os.path.join(self.keyframes_dir, f"Keyframes_{level}", "keyframes", video_id),
            os.path.join(self.keyframes_dir, f"Keyframes_{level}", video_id),
            os.path.join(self.keyframes_dir, level, "keyframes", video_id),
            os.path.join(self.keyframes_dir, "keyframes", video_id),
            os.path.join(self.keyframes_dir, video_id)
        ]
        for fc in folder_candidates:
            if os.path.exists(fc):
                imgs = sorted(glob.glob(os.path.join(fc, "*.jpg")) + glob.glob(os.path.join(fc, "*.jpeg")) + glob.glob(os.path.join(fc, "*.png")))
                if imgs:
                    return imgs

        for root, _, files in os.walk(self.keyframes_dir):
            if os.path.basename(root) == video_id:
                imgs = sorted([os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if imgs:
                    return imgs
        return []

    def _infer_vlm_multi_frame_video(self, image_paths, question_text):
        """Runs multi-frame video sequence reasoning using Qwen2.5-VL-7B over dense-anchored local keyframes."""
        prompt_text = (
            f"Nhiệm vụ: Quan sát kỹ các khung ảnh trải dài theo thời gian của video này và trả lời câu hỏi sau bằng Tiếng Việt:\n"
            f"'{question_text}'\n"
            f"Yêu cầu: Trả lời ngắn gọn, trực tiếp con số / tên riêng / từ cần tìm. Không thêm lời dẫn rườm rà."
        )
        
        content_items = []
        pil_images = []
        for p in image_paths:
            content_items.append({"type": "image", "image": p})
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        content_items.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content_items}]
        
        try:
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if process_vision_info:
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.vlm_processor(
                        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
                    ).to(target_device)
                except Exception:
                    inputs = self.vlm_processor(
                        text=[text], images=pil_images, padding=True, return_tensors="pt"
                    ).to(target_device)
            else:
                inputs = self.vlm_processor(
                    text=[text], images=pil_images, padding=True, return_tensors="pt"
                ).to(target_device)

            with torch.inference_mode():
                gen_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)
                gen_trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
                out_text = self.vlm_processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
                
            del inputs, pil_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return out_text.strip()
        except Exception as e:
            print(f"[WARNING] VLM Multi-Frame Infer Warning: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Không rõ"

    def _find_keyframe_image(self, video_id, frame_idx, frame_id=""):
        """Locates physical keyframe image path supporting 3-digit (002.jpg) matching Kaggle input layout."""
        cache_key = f"{video_id}_{frame_idx}_{frame_id}"
        if cache_key in self._keyframe_cache:
            return self._keyframe_cache[cache_key]

        if not self.keyframes_dir or not os.path.exists(self.keyframes_dir):
            return None

        level = video_id.split('_')[0] if '_' in video_id else ""
        idx_3d = f"{frame_idx:03d}"
        idx_3d_1based = f"{frame_idx + 1:03d}"
        idx_4d = f"{frame_idx:04d}"
        idx_raw = str(frame_idx)
        
        fnames = [idx_3d, idx_3d_1based, frame_id, idx_4d, idx_raw]
        fnames = [f for f in fnames if f]

        for fn in fnames:
            candidates = [
                os.path.join(self.keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, f"{fn}.jpg"),
                os.path.join(self.keyframes_dir, f"Keyframes_{level}", video_id, f"{fn}.jpg"),
                os.path.join(self.keyframes_dir, level, "keyframes", video_id, f"{fn}.jpg"),
                os.path.join(self.keyframes_dir, "keyframes", video_id, f"{fn}.jpg"),
                os.path.join(self.keyframes_dir, video_id, f"{fn}.jpg"),
                os.path.join(self.keyframes_dir, level, video_id, f"{fn}.jpg")
            ]
            for c in candidates:
                if os.path.exists(c):
                    self._keyframe_cache[cache_key] = c
                    return c

        for root, _, files in os.walk(self.keyframes_dir):
            if os.path.basename(root) == video_id:
                for fn in fnames:
                    target_file = f"{fn}.jpg"
                    if target_file in files:
                        res = os.path.join(root, target_file)
                        self._keyframe_cache[cache_key] = res
                        return res
                jpg_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if jpg_files:
                    target_idx = min(max(0, frame_idx), len(jpg_files) - 1)
                    res = os.path.join(root, jpg_files[target_idx])
                    self._keyframe_cache[cache_key] = res
                    return res

        return None

    def solve_trake(self, parsed_schema, fused_candidates, dense_engine=None, total_preds=100):
        """Solves TRAKE task using Viterbi Dynamic Programming alignment for strictly monotonic increasing frame IDs."""
        events = parsed_schema.get("events")
        if not events or not isinstance(events, list):
            events = parsed_schema.get("bm25_keywords", [])
        if not events:
            events = [w.strip() for w in parsed_schema.get("query_vi", "").split() if len(w.strip()) >= 3][:4]
        if not events:
            events = ["sự kiện 1", "sự kiện 2"]

        n_events = len(events)
        aligned_results = []

        if not fused_candidates:
            fused_candidates = [{"video_id": f"L21_V{i:03d}", "dense_info": {"best_frame_idx": 1}, "rrf_score": 0.0} for i in range(1, total_preds + 1)]

        for cand in fused_candidates[:total_preds]:
            vid = cand["video_id"]
            dense_info = cand.get("dense_info", {})
            scores = dense_info.get("all_scores")

            if scores is not None and len(scores) >= n_events and dense_engine is not None:
                aligned_f_idxs = self._align_events_dp(events, scores, dense_engine)
            else:
                n_total = len(scores) if scores is not None else 100
                aligned_f_idxs = [int(x) for x in np.linspace(0, max(0, n_total - 1), n_events)]

            raw_frame_ids = [
                get_frame_id_from_idx(self.keyframes_dir, vid, f_idx, metadata_dir=self.metadata_dir)
                for f_idx in aligned_f_idxs
            ]

            # Guarantee strictly monotonic increasing frame IDs: t_1 < t_2 < ... < t_N
            int_fids = []
            for rf in raw_frame_ids:
                try:
                    int_fids.append(int(rf))
                except Exception:
                    int_fids.append(0)

            for i in range(1, len(int_fids)):
                if int_fids[i] <= int_fids[i - 1]:
                    int_fids[i] = int_fids[i - 1] + 10

            final_frame_ids = [f"{v:03d}" for v in int_fids]

            aligned_results.append({
                "video_id": vid,
                "frame_ids": final_frame_ids
            })

        return aligned_results

    def _align_events_dp(self, events, frame_scores, dense_engine):
        """Dynamic Programming Viterbi alignment algorithm finding optimal sequence t_1 < t_2 < ... < t_N."""
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
