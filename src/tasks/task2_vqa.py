import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from src.tasks.task1_kis import (
    gaussian_smooth_scores,
    generate_diversity_top100_kis,
    get_frame_id_from_idx,
)
from src.utils import get_keyframe_path_by_index, FrameOCRStore


_vlm_model = None
_vlm_processor = None


def load_vlm(model_id="Qwen/Qwen2-VL-2B-Instruct"):
    """Nap mot VLM singleton, uu tien GPU thu hai neu co."""
    global _vlm_model, _vlm_processor
    if _vlm_model is not None and _vlm_processor is not None:
        return _vlm_model, _vlm_processor

    print("VLM: Nap mo hinh %s (shared singleton)..." % model_id)
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            device_map = {"": "cuda:1"}
            print("VLM: Dual-GPU -> dat VLM tren cuda:1.")
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
            device_map=device_map,
        )
    except Exception:
        from transformers import AutoModelForVision2Seq

        _vlm_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    try:
        _vlm_processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except Exception:
        _vlm_processor = AutoProcessor.from_pretrained(model_id)
    print("VLM: Khoi tao thanh cong.")
    return _vlm_model, _vlm_processor


def classify_qa_type(question):
    """Phan loai bon dang Q&A chinh thuc ma khong hardcode query ID."""
    text = str(question or "").lower()
    if any(phrase in text for phrase in ("có bao nhiêu", "bao nhiêu vị trí", "đếm")):
        return "counting"
    if any(phrase in text for phrase in ("tên của", "tên con", "địa danh", "đèo là gì")):
        return "place_name"
    if any(phrase in text for phrase in ("biển báo", "biển số", "con số được ghi")):
        return "sign_number"
    if any(phrase in text for phrase in ("con số hiển thị", "số hiển thị", "trên cân")):
        return "read_number"
    if any(phrase in text for phrase in ("chữ gì", "ghi gì", "đọc", "tên gì")):
        return "read_text"
    return "generic"


def clean_vlm_answer(raw_answer, question=None):
    """Lam sach answer nhung khong tu bien reasoning thanh dap an gia."""
    if raw_answer is None:
        return "Không rõ"
    answer = str(raw_answer).strip()
    if not answer:
        return "Không rõ"

    if question and len(question) > 10:
        clean_question = question.strip().rstrip("?")
        if answer.lower().startswith(clean_question.lower()):
            answer = answer[len(clean_question):].lstrip(" : là,.-")

    prefixes = [
        r"^(?:đáp án|câu trả lời|kết quả)\s*(?:là)?\s*[:\-\.]?\s*",
        r"^(?:đó là|chính là|là)\s*[:\-\.]?\s*",
        r"^(?:trên|trong)\s+(?:hình|ảnh)\s*(?:là|có)?\s*[:\-\.]?\s*",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in prefixes:
            updated = re.sub(pattern, "", answer, flags=re.IGNORECASE).strip()
            if updated and updated != answer:
                answer = updated
                changed = True

    answer = answer.strip().strip('"').strip("'").rstrip(".!?;:")
    refusal_terms = (
        "không rõ",
        "chưa rõ",
        "không thể xác định",
        "không xác định",
        "không nhìn thấy",
        "không thấy",
        "xin lỗi",
        "tôi không",
        "none",
    )
    if not answer or any(term in answer.lower() for term in refusal_terms):
        return "Không rõ"
    return answer[:100].strip() or "Không rõ"


def _normalize_answer(value):
    text = str(value or "").lower().strip()
    text = re.sub(r"[^0-9a-zà-ỹđ]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _is_concrete_answer(answer):
    return _normalize_answer(answer) not in {
        "",
        "không rõ",
        "chưa rõ",
        "none",
        "không có",
    }


def _answer_matches_type(answer, qa_type):
    if not _is_concrete_answer(answer):
        return False
    text = _normalize_answer(answer)
    if qa_type in {"read_number", "sign_number", "counting"}:
        if re.search(r"\d", text):
            return True
        number_words = {
            "không", "một", "hai", "ba", "bốn", "tư", "năm", "lăm",
            "sáu", "bảy", "tám", "chín", "mười", "trăm", "nghìn",
        }
        return any(token in number_words for token in text.split())
    if qa_type in {"place_name", "read_text"}:
        return bool(re.search(r"[a-zà-ỹđ]{2,}", text, flags=re.IGNORECASE))
    return True



def _ocr_quality(text, qa_type):
    text = str(text or "").strip()
    if not text:
        return 0.0
    digits = re.findall(r"\d+(?:[.,]\d+)?", text)
    letters = re.findall(r"[A-Za-zÀ-ỹĐđ]{2,}", text)
    if qa_type in {"read_number", "sign_number"}:
        return min(1.0, 0.25 + 0.25 * len(digits)) if digits else 0.1
    if qa_type in {"place_name", "read_text"}:
        return min(1.0, len(letters) / 6.0)
    if qa_type == "counting":
        # OCR chi ho tro doc legend/nhan; khong dung so OCR lam ket qua dem.
        return min(0.5, (len(digits) + len(letters)) / 12.0)
    return min(0.5, (len(digits) + len(letters)) / 12.0)


def select_evidence_keyframes(
    dense_info,
    ocr_by_ordinal,
    qa_type,
    total_budget=6,
    dense_weight=0.65,
    ocr_weight=0.35,
    temporal_nms_distance=1,
):
    """Xep hang frame evidence bang semantic score + frame-level OCR."""
    if not dense_info or dense_info.get("all_scores") is None:
        return [{
            "keyframe_ordinal": 0,
            "evidence_score": 0.0,
            "dense_score": None,
            "ocr_score": _ocr_quality(ocr_by_ordinal.get(0, ""), qa_type),
            "ocr_text": ocr_by_ordinal.get(0, ""),
        }]

    scores = np.asarray(dense_info["all_scores"], dtype=np.float32).reshape(-1)
    n_frames = len(scores)
    if n_frames == 0:
        return []
    clean = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    smoothed = np.asarray(gaussian_smooth_scores(clean, sigma=1.5)).reshape(-1)
    if len(smoothed) != n_frames:
        smoothed = clean
    low = float(np.min(smoothed))
    high = float(np.max(smoothed))
    dense_normalized = (
        np.ones(n_frames, dtype=np.float32)
        if high - low <= 1e-8
        else (smoothed - low) / (high - low)
    )

    candidate_indices = set(int(index) for index in np.argsort(smoothed)[::-1][: min(24, n_frames)])
    global_peak = int(np.argmax(smoothed))
    for delta in (-2, -1, 0, 1, 2):
        index = global_peak + delta
        if 0 <= index < n_frames:
            candidate_indices.add(index)
    candidate_indices.update(
        int(index) for index in ocr_by_ordinal.keys() if 0 <= int(index) < n_frames
    )

    ranked = []
    for index in candidate_indices:
        ocr_text = ocr_by_ordinal.get(index, "")
        ocr_score = _ocr_quality(ocr_text, qa_type)
        evidence_score = (
            max(0.0, float(dense_weight)) * float(dense_normalized[index])
            + max(0.0, float(ocr_weight)) * float(ocr_score)
        )
        weight_sum = max(1e-8, max(0.0, dense_weight) + max(0.0, ocr_weight))
        ranked.append({
            "keyframe_ordinal": int(index),
            "evidence_score": float(evidence_score / weight_sum),
            "dense_score": float(smoothed[index]),
            "ocr_score": float(ocr_score),
            "ocr_text": ocr_text,
        })
    ranked.sort(key=lambda item: item["evidence_score"], reverse=True)

    selected = []
    min_distance = max(0, int(temporal_nms_distance))
    for item in ranked:
        index = item["keyframe_ordinal"]
        if all(abs(index - previous["keyframe_ordinal"]) > min_distance for previous in selected):
            selected.append(item)
        if len(selected) >= max(1, int(total_budget)):
            break
    return selected


def _instruction_for_type(qa_type):
    if qa_type == "read_number":
        return (
            "Đọc đúng con số ở trạng thái cuối/ổn định được hỏi. "
            "Ưu tiên vùng màn hình cân; không lấy số từ phụ đề hoặc logo."
        )
    if qa_type == "sign_number":
        return (
            "Đọc đúng con số trên biển báo được mô tả. "
            "Phân biệt biển báo với số xe, phụ đề và watermark."
        )
    if qa_type == "counting":
        return (
            "Đếm trên frame rõ nhất. Các ảnh là những thời điểm/góc nhìn của cùng video: "
            "không cộng số lượng giữa các ảnh và không đếm bảng chú giải."
        )
    if qa_type == "place_name":
        return (
            "Tìm chữ/tên địa danh xuất hiện trên biển, cột mốc, bản đồ hoặc phụ đề liên quan. "
            "Đối chiếu OCR gợi ý với hình ảnh trước khi trả lời."
        )
    return "Trả lời từ frame có bằng chứng trực tiếp và chọn đúng evidence image."


def _parse_vlm_json(raw_text, question, image_count):
    raw_text = str(raw_text or "").strip()
    payload = None
    matches = re.findall(r"\{.*?\}", raw_text, flags=re.DOTALL)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return {
            "answer": clean_vlm_answer(raw_text, question=question),
            "evidence_image": 1 if image_count else None,
            "confidence": None,
            "evidence_source": "vlm_unstructured",
            "raw_answer": raw_text,
        }

    answer = clean_vlm_answer(payload.get("answer"), question=question)
    try:
        evidence_image = int(payload.get("evidence_image"))
    except (TypeError, ValueError):
        evidence_image = 1 if image_count else None
    if evidence_image is not None and not (1 <= evidence_image <= image_count):
        evidence_image = 1 if image_count else None

    confidence = payload.get("confidence")
    try:
        confidence = float(confidence)
        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError, OverflowError):
        confidence = None

    source = str(payload.get("evidence_source") or "vlm_visual").strip()
    return {
        "answer": answer,
        "evidence_image": evidence_image,
        "confidence": confidence,
        "evidence_source": source[:80] or "vlm_visual",
        "raw_answer": raw_text,
    }


def solve_single_video_vqa(
    video_id,
    dense_info,
    query_text,
    question,
    keyframes_dir,
    model,
    processor,
    metadata_dir=None,
    ocr_by_ordinal=None,
    qa_type="generic",
    qa_config=None,
):
    qa_config = qa_config or {}
    ocr_by_ordinal = ocr_by_ordinal or {}
    evidence_candidates = select_evidence_keyframes(
        dense_info=dense_info,
        ocr_by_ordinal=ocr_by_ordinal,
        qa_type=qa_type,
        total_budget=qa_config.get("evidence_frame_budget", 6),
        dense_weight=qa_config.get("evidence_dense_weight", 0.65),
        ocr_weight=qa_config.get("evidence_ocr_weight", 0.35),
        temporal_nms_distance=qa_config.get("temporal_nms_distance", 1),
    )

    usable = []
    for item in evidence_candidates:
        ordinal = int(item["keyframe_ordinal"])
        try:
            image_path = get_keyframe_path_by_index(
                keyframes_dir,
                video_id,
                ordinal,
            )
            actual_frame = get_frame_id_from_idx(
                keyframes_dir,
                video_id,
                ordinal,
                metadata_dir=metadata_dir,
            )
        except (IndexError, FileNotFoundError, ValueError, RuntimeError):
            continue
        if image_path and os.path.isfile(image_path):
            enriched = dict(item)
            enriched["image_path"] = image_path
            enriched["frame_id"] = str(actual_frame)
            usable.append(enriched)

    if not usable:
        return {
            "video_id": video_id,
            "frame_id": None,
            "answer": "Không rõ",
            "confidence": None,
            "evidence_source": "no_evidence_frame",
            "evidence_score": 0.0,
            "has_concrete_answer": False,
            "evidence_candidates": [],
        }

    content = []
    for image_number, item in enumerate(usable, start=1):
        ocr_hint = item.get("ocr_text") or "(không có OCR precomputed)"
        content.append({
            "type": "text",
            "text": (
                "EVIDENCE_IMAGE_%d | OCR_HINT: %s"
                % (image_number, ocr_hint[:500])
            ),
        })
        content.append({"type": "image", "image": item["image_path"]})

    prompt = (
        "Bối cảnh cần tìm: %s\n"
        "Câu hỏi: %s\n"
        "Loại câu hỏi: %s\n"
        "%s\n"
        "Hãy đối chiếu tất cả ảnh nhưng chỉ chọn ảnh có bằng chứng trực tiếp nhất. "
        "OCR_HINT có thể sai, chỉ dùng khi khớp hình ảnh. "
        "Nếu không đủ bằng chứng, answer phải là 'Không rõ'.\n"
        "Trả về DUY NHẤT JSON hợp lệ, không markdown, ví dụ đúng định dạng "
        "(không copy nguyên văn các giá trị mẫu dưới đây):\n"
        "{\"answer\": \"<đáp án ngắn dưới 20 từ>\", "
        "\"evidence_image\": <số thứ tự ảnh>, \"confidence\": <0.0-1.0>, "
        "\"evidence_source\": \"<ocr hoặc visual hoặc ocr+visual>\"}"
        % (query_text, question, qa_type, _instruction_for_type(qa_type))
    )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    raw_output = ""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=int(qa_config.get("max_new_tokens", 120)),
                do_sample=False,
            )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    except Exception as exc:
        print("VQA warning video=%s: %s" % (video_id, exc))
        raw_output = ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parsed = _parse_vlm_json(raw_output, question, len(usable))
    print("  [DEBUG] raw_output=%r" % raw_output[:300])
    evidence_index = (parsed.get("evidence_image") or 1) - 1
    evidence_index = max(0, min(len(usable) - 1, evidence_index))
    chosen = usable[evidence_index]
    has_ocr = bool(chosen.get("ocr_text"))
    source = parsed["evidence_source"]
    if has_ocr and "ocr" not in source.lower():
        source = "ocr_available+" + source

    return {
        "video_id": video_id,
        "frame_id": chosen["frame_id"],
        "coarse_evidence_frame_id": chosen["frame_id"],
        "evidence_keyframe_ordinal": chosen["keyframe_ordinal"],
        "evidence_image_path": chosen["image_path"],
        "answer": parsed["answer"],
        "raw_answer": parsed["raw_answer"],
        "confidence": parsed["confidence"],
        "evidence_source": source,
        "evidence_score": float(chosen["evidence_score"]),
        "has_concrete_answer": _is_concrete_answer(parsed["answer"]),
        "answer_matches_type": _answer_matches_type(parsed["answer"], qa_type),
        "evidence_candidates": [
            {
                "image_number": index + 1,
                "keyframe_ordinal": item["keyframe_ordinal"],
                "frame_id": item["frame_id"],
                "evidence_score": item["evidence_score"],
                "dense_score": item["dense_score"],
                "ocr_score": item["ocr_score"],
                "ocr_text": item["ocr_text"],
            }
            for index, item in enumerate(usable)
        ],
    }


def _candidate_quality(result, original_rank, qa_config):
    if not result.get("answer_matches_type", result.get("has_concrete_answer")):
        return -1.0
    retrieval_score = 1.0 / (1.0 + max(0, int(original_rank)))
    evidence_score = float(result.get("evidence_score", 0.0) or 0.0)
    confidence = result.get("confidence")
    confidence_score = float(confidence) if confidence is not None else 0.5
    weights = {
        "retrieval": float(qa_config.get("candidate_retrieval_weight", 0.40)),
        "evidence": float(qa_config.get("candidate_evidence_weight", 0.30)),
        "confidence": float(qa_config.get("candidate_confidence_weight", 0.30)),
    }
    denominator = sum(max(0.0, value) for value in weights.values()) or 1.0
    return float((
        weights["retrieval"] * retrieval_score
        + weights["evidence"] * evidence_score
        + weights["confidence"] * confidence_score
    ) / denominator)


def _refine_candidate_evidence(
    evaluated_results,
    query_id,
    query_text,
    question,
    fused_candidates,
    temporal_refiner,
    query_processor,
):
    if not evaluated_results or temporal_refiner is None or query_processor is None:
        return None
    localization_query = "%s %s" % (query_text, question)
    localization_info = query_processor.process(localization_query)
    coarse = [
        {"video_id": item["video_id"], "frame_id": item["frame_id"]}
        for item in evaluated_results
        if item.get("frame_id") is not None
    ]
    if not coarse:
        return None
    refined, trace = temporal_refiner.refine_kis_predictions(
        query_id=(str(query_id or "qa") + "_qa_evidence"),
        query_text=localization_query,
        prompt_ensemble=localization_info["prompt_ensemble"],
        coarse_predictions=coarse,
        fused_candidates=fused_candidates,
        query_processor=None,
    )
    refined_lookup = {
        str(item["video_id"]): str(item["frame_id"])
        for item in refined
    }
    trace_lookup = {
        str(item.get("video_id")): item
        for item in trace.get("refinements", [])
    }
    for result in evaluated_results:
        video_id = str(result["video_id"])
        if video_id in refined_lookup:
            result["frame_id"] = refined_lookup[video_id]
            refinement = trace_lookup.get(video_id, {})
            result["refinement_score"] = refinement.get("score")
            result["refinement_status"] = refinement.get("status")
            if refinement.get("status") == "refined":
                result["evidence_source"] += "+raw_clip_refine"
    return trace


def solve_task2(
    query_text,
    question,
    fused_candidates,
    keyframes_dir,
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    metadata_dir=None,
    object_searcher=None,
    ocr_dir=None,
    qa_config=None,
    temporal_refiner=None,
    query_processor=None,
    query_id=None,
):
    """Q&A = video retrieval -> evidence frames -> answer -> evidence frame."""
    del object_searcher  # Giu interface caller cu; Object boost da chay truoc do.
    qa_config = qa_config or {}
    if not fused_candidates:
        return {
            "video_id": "none",
            "frame_id": None,
            "answer": "Không rõ",
            "promoted_idx": 0,
            "candidate_results": [],
        }

    qa_type = classify_qa_type(question)
    model, processor = load_vlm(model_id)
    ocr_store = FrameOCRStore(ocr_dir, keyframes_dir)
    candidate_count = min(
        len(fused_candidates),
        max(1, int(qa_config.get("candidate_video_count", 5))),
    )
    evaluated_results = []
    for original_rank, candidate in enumerate(fused_candidates[:candidate_count]):
        video_id = str(candidate["video_id"])
        print("VQA: candidate #%d %s" % (original_rank + 1, video_id))
        result = solve_single_video_vqa(
            video_id=video_id,
            dense_info=candidate.get("dense_info"),
            query_text=query_text,
            question=question,
            keyframes_dir=keyframes_dir,
            model=model,
            processor=processor,
            metadata_dir=metadata_dir,
            ocr_by_ordinal=ocr_store.get_by_ordinal(video_id),
            qa_type=qa_type,
            qa_config=qa_config,
        )
        result["original_rank"] = original_rank
        result["promoted_idx"] = original_rank
        result["qa_type"] = qa_type
        result["quality_score"] = _candidate_quality(
            result,
            original_rank,
            qa_config,
        )
        evaluated_results.append(result)
        print(
            "  answer='%s' evidence_frame=%s source=%s score=%.4f"
            % (
                result["answer"],
                result["frame_id"],
                result["evidence_source"],
                result["quality_score"],
            )
        )

    temporal_trace = _refine_candidate_evidence(
        evaluated_results=evaluated_results,
        query_id=query_id,
        query_text=query_text,
        question=question,
        fused_candidates=fused_candidates,
        temporal_refiner=temporal_refiner,
        query_processor=query_processor,
    )
    evaluated_results.sort(key=lambda item: item["quality_score"], reverse=True)
    best = dict(evaluated_results[0])
    best["candidate_results"] = evaluated_results
    best["temporal_refinement_trace"] = temporal_trace
    return best


def _consensus_answer(candidate_results, qa_config):
    threshold = float(qa_config.get("consensus_confidence_threshold", 0.65))
    minimum = max(2, int(qa_config.get("consensus_min_candidates", 2)))
    groups = {}
    for item in candidate_results:
        if not item.get("answer_matches_type", item.get("has_concrete_answer")):
            continue
        confidence = item.get("confidence")
        if confidence is not None and float(confidence) < threshold:
            continue
        key = _normalize_answer(item.get("answer"))
        if key:
            groups.setdefault(key, []).append(item)
    if not groups:
        return "Không rõ", False
    _, group = max(groups.items(), key=lambda pair: len(pair[1]))
    if len(group) < minimum:
        return "Không rõ", False
    return str(group[0]["answer"]), True


def build_task2_top100_predictions(
    fused_candidates,
    answer_result,
    keyframes_dir,
    metadata_dir=None,
    total_preds=100,
    qa_config=None,
):
    """Tao Q&A Top-100 voi answer/evidence frame theo tung candidate video."""
    qa_config = qa_config or {}
    coarse = generate_diversity_top100_kis(
        fused_candidates,
        keyframes_dir,
        metadata_dir=metadata_dir,
        total_preds=total_preds,
    )
    candidate_results = answer_result.get("candidate_results") or []
    result_lookup = {
        str(item.get("video_id")): item
        for item in candidate_results
    }
    consensus_answer, has_consensus = _consensus_answer(
        candidate_results,
        qa_config,
    )
    first_video_occurrence = set()
    displaced_frame_by_video = {}
    output = []
    emitted = set()
    for prediction in coarse:
        video_id = str(prediction.get("video_id"))
        coarse_frame_id = str(prediction.get("frame_id"))
        frame_id = coarse_frame_id
        result = result_lookup.get(video_id)
        if result is not None:
            answer = result.get("answer", "Không rõ")
            if video_id not in first_video_occurrence and result.get("frame_id") is not None:
                evidence_frame_id = str(result["frame_id"])
                if evidence_frame_id != coarse_frame_id:
                    frame_id = evidence_frame_id
                    # Neu evidence frame da ton tai o mot slot coarse phia sau,
                    # slot do se nhan lai frame vua bi thay the. Cach swap nay
                    # bao toan so luong Top-100 ma khong lap tuple.
                    displaced_frame_by_video[video_id] = coarse_frame_id
            source = result.get("evidence_source", "vlm_visual")
            score = result.get("evidence_score")
            confidence = result.get("confidence")
        elif has_consensus:
            answer = consensus_answer
            source = "cross_candidate_consensus"
            score = None
            confidence = None
        else:
            answer = "Không rõ"
            source = "unevaluated_no_consensus"
            score = None
            confidence = None

        pair = (video_id, frame_id, _normalize_answer(answer))
        if pair in emitted:
            displaced = displaced_frame_by_video.pop(video_id, None)
            if displaced is not None:
                frame_id = displaced
            else:
                frame_id = coarse_frame_id
            pair = (video_id, frame_id, _normalize_answer(answer))
        if pair in emitted:
            continue
        emitted.add(pair)
        first_video_occurrence.add(video_id)
        output.append({
            "video_id": video_id,
            "frame_id": frame_id,
            "answer": answer,
            "evidence_source": source,
            "evidence_score": score,
            "confidence": confidence,
        })
        if len(output) >= min(100, int(total_preds)):
            break
    return output
