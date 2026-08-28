import re
import unicodedata
from collections import defaultdict

import numpy as np

from src.utils import normalize_query_item


K_VALUES = (1, 5, 20, 50, 100)
VIDEO_K_VALUES = (1, 5, 20)
FRAME_K_VALUES = (1, 5, 20)


def _safe_int(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _normalize_answer_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.strip('"\'`“”‘’')
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


NUMBER_ALIASES = {
    "không": "0",
    "zero": "0",
    "một": "1",
    "one": "1",
    "hai": "2",
    "two": "2",
    "ba": "3",
    "three": "3",
    "bốn": "4",
    "tư": "4",
    "four": "4",
    "năm": "5",
    "five": "5",
    "sáu": "6",
    "six": "6",
    "bảy": "7",
    "seven": "7",
    "tám": "8",
    "eight": "8",
    "chín": "9",
    "nine": "9",
    "mười": "10",
    "ten": "10",
}


class Evaluator:
    """
    Bộ đánh giá local cho AIC 2026.

    - R@k bám theo công thức BTC: max R-Score trong Top-k.
    - Video Recall dùng danh sách video retrieval riêng nếu được cung cấp.
    - Frame Recall luôn yêu cầu đúng video và đúng frame/alignment, nhưng bỏ
      qua answer của Q&A.
    - Không dùng metric local để khẳng định chất lượng nếu Ground Truth là
      placeholder/smoke-test.
    """

    def __init__(self, ground_truth, ground_truth_source=None):
        self.raw_ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
        self.ground_truth_source = str(ground_truth_source or "")
        self.gt = {"task1": {}, "task2": {}, "task3": {}}

        for task_type in self.gt:
            items = self.raw_ground_truth.get(task_type, [])
            if isinstance(items, dict):
                iterator = items.items()
            elif isinstance(items, list):
                iterator = [(None, item) for item in items]
            else:
                iterator = []

            for fallback_id, item in iterator:
                norm_item = normalize_query_item(item)
                if norm_item["query_id"] == "unknown" and fallback_id is not None:
                    norm_item["query_id"] = str(fallback_id)
                self.gt[task_type][norm_item["query_id"]] = norm_item

        self.ground_truth_report = self._inspect_ground_truth()

    def _inspect_ground_truth(self):
        issues = []
        counts = {task: len(items) for task, items in self.gt.items()}
        source_name = self.ground_truth_source.replace("\\", "/").lower()
        source_name = "/" + source_name.lstrip("/")

        meta = self.raw_ground_truth.get("_meta", {})
        if not isinstance(meta, dict):
            meta = {}

        explicit_status = str(
            meta.get("status")
            or meta.get("ground_truth_status")
            or self.raw_ground_truth.get("ground_truth_status")
            or ""
        ).lower()

        placeholder_reasons = []
        if explicit_status in {
            "placeholder",
            "sample",
            "smoke_test",
            "demo",
            "template",
            "annotation_required",
        }:
            placeholder_reasons.append("Ground Truth được đánh dấu là dữ liệu mẫu.")

        if any(
            token in source_name
            for token in (
                "/src/label/sample.json",
                "/src/label/task1.json",
                "/src/label/task2.json",
                "/src/label/task3.json",
            )
        ):
            placeholder_reasons.append("Đường dẫn Ground Truth thuộc bộ label mẫu của project.")

        for task_type, task_items in self.gt.items():
            for query_id, item in task_items.items():
                prefix = "%s/%s" % (task_type, query_id)
                if not item.get("video_id"):
                    issues.append("%s: thiếu video_id" % prefix)

                if task_type in {"task1", "task2"}:
                    start = _safe_int(item.get("frame_start"))
                    end = _safe_int(item.get("frame_end"))
                    if start is None or end is None or start > end:
                        issues.append("%s: frame range không hợp lệ" % prefix)

                if task_type == "task2" and not item.get("answer"):
                    issues.append("%s: thiếu answer" % prefix)

                if task_type == "task3":
                    events = item.get("events_dicts") or []
                    if not events:
                        issues.append("%s: thiếu events" % prefix)
                    for event_idx, event in enumerate(events, start=1):
                        start = _safe_int(event.get("frame_start"))
                        end = _safe_int(event.get("frame_end"))
                        if start is None or end is None or start > end:
                            issues.append(
                                "%s: event %d có frame range không hợp lệ"
                                % (prefix, event_idx)
                            )

        return {
            "source": self.ground_truth_source or None,
            "counts": counts,
            "total_queries": sum(counts.values()),
            "is_placeholder": bool(placeholder_reasons),
            "placeholder_reasons": placeholder_reasons,
            "issues": issues,
            "is_usable": sum(counts.values()) > 0 and not issues,
        }

    def _acceptable_answers(self, gt_item):
        raw = gt_item.get("raw", {})
        values = [gt_item.get("answer", "")]
        if isinstance(raw, dict):
            for key in ("acceptable_answers", "answers", "answer_aliases"):
                extra = raw.get(key, [])
                if isinstance(extra, (str, int, float)):
                    values.append(extra)
                elif isinstance(extra, list):
                    values.extend(extra)
        return values

    def check_answer_match(self, prediction, gt_item):
        """
        So khớp answer theo hướng bảo thủ.

        Không dùng token-overlap vì "màu đỏ" và "màu xanh" cùng có từ
        "màu" nhưng là hai đáp án khác nhau. Dev-set có thể khai báo
        acceptable_answers để chấp nhận các cách diễn đạt tương đương.
        """
        pred = _normalize_answer_text(prediction)
        if not pred:
            return False

        pred_canonical = NUMBER_ALIASES.get(pred, pred)
        for answer in self._acceptable_answers(gt_item):
            gt = _normalize_answer_text(answer)
            if not gt:
                continue
            if pred_canonical == NUMBER_ALIASES.get(gt, gt):
                return True
        return False

    def _score_prediction(self, task_type, prediction, gt_item):
        pred = prediction if isinstance(prediction, dict) else {}
        gt_video = str(gt_item.get("video_id", ""))
        pred_video = str(pred.get("video_id", ""))
        video_match = bool(gt_video) and pred_video == gt_video

        result = {
            "video_match": video_match,
            "frame_match": False,
            "answer_match": None,
            "r_score": 0.0,
            "event_matches": [],
            "correct_event_count": 0,
            "event_accuracy": None,
            "temporal_order_error": False,
            "schema_valid": True,
        }

        if task_type in {"task1", "task2"}:
            frame_id = _safe_int(pred.get("frame_id"))
            start = _safe_int(gt_item.get("frame_start"))
            end = _safe_int(gt_item.get("frame_end"))
            result["schema_valid"] = frame_id is not None
            if video_match and frame_id is not None and start is not None and end is not None:
                result["frame_match"] = start <= frame_id <= end

            if task_type == "task1":
                result["r_score"] = 1.0 if result["frame_match"] else 0.0
            else:
                result["answer_match"] = self.check_answer_match(
                    pred.get("answer", ""), gt_item
                )
                result["r_score"] = (
                    1.0
                    if result["frame_match"] and result["answer_match"]
                    else 0.0
                )
            return result

        if task_type == "task3":
            events = gt_item.get("events_dicts") or []
            raw_frames = pred.get("frame_ids", [])
            if not isinstance(raw_frames, (list, tuple)):
                raw_frames = []
            frames = [_safe_int(value) for value in raw_frames]
            result["schema_valid"] = (
                bool(events)
                and len(frames) == len(events)
                and all(frame is not None for frame in frames)
            )

            valid_frames = [frame for frame in frames if frame is not None]
            if len(valid_frames) == len(frames) and len(frames) > 1:
                result["temporal_order_error"] = any(
                    frames[idx] >= frames[idx + 1]
                    for idx in range(len(frames) - 1)
                )

            event_matches = []
            for event_idx, event in enumerate(events):
                match = False
                if video_match and event_idx < len(frames):
                    frame_id = frames[event_idx]
                    start = _safe_int(event.get("frame_start"))
                    end = _safe_int(event.get("frame_end"))
                    if frame_id is not None and start is not None and end is not None:
                        match = start <= frame_id <= end
                event_matches.append(match)

            result["event_matches"] = event_matches
            result["correct_event_count"] = sum(event_matches)
            result["event_accuracy"] = (
                result["correct_event_count"] / len(events) if events else 0.0
            )
            result["frame_match"] = bool(events) and all(event_matches)
            result["r_score"] = result["event_accuracy"] if video_match else 0.0
            return result

        result["schema_valid"] = False
        return result

    @staticmethod
    def _find_video_rank(candidates, gt_video):
        seen = set()
        unique_rank = 0
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            video_id = str(candidate.get("video_id", ""))
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            unique_rank += 1
            if video_id == gt_video:
                return unique_rank
        return None

    def evaluate_query_detailed(
        self,
        query_id,
        task_type,
        predictions,
        retrieval_candidates=None,
    ):
        if query_id not in self.gt.get(task_type, {}):
            raise KeyError("Không tìm thấy Ground Truth cho %s/%s" % (task_type, query_id))

        gt_item = self.gt[task_type][query_id]
        preds = list(predictions or [])[:100]
        scored = [self._score_prediction(task_type, pred, gt_item) for pred in preds]

        retrieval_source = "retrieval_trace"
        video_candidates = retrieval_candidates
        if video_candidates is None:
            video_candidates = preds
            retrieval_source = "submission_unique_videos_fallback"

        video_rank = self._find_video_rank(video_candidates, gt_item["video_id"])

        frame_rank = next(
            (idx + 1 for idx, item in enumerate(scored) if item["frame_match"]),
            None,
        )
        correct_rank = next(
            (idx + 1 for idx, item in enumerate(scored) if item["r_score"] >= 1.0),
            None,
        )

        r_scores = [item["r_score"] for item in scored]
        r_scores.extend([0.0] * (100 - len(r_scores)))
        r_at_k = {
            "R@%d" % k: float(max(r_scores[:k])) if r_scores[:k] else 0.0
            for k in K_VALUES
        }

        video_recall = {
            "Video Recall@%d" % k: float(video_rank is not None and video_rank <= k)
            for k in VIDEO_K_VALUES
        }
        frame_recall = {
            "Frame Recall@%d" % k: float(frame_rank is not None and frame_rank <= k)
            for k in FRAME_K_VALUES
        }

        top1 = scored[0] if scored else None
        top1_failure_type = "no_prediction"
        if top1 is not None:
            if not top1["video_match"]:
                top1_failure_type = "wrong_video"
            elif not top1["frame_match"]:
                top1_failure_type = (
                    "alignment_error" if task_type == "task3" else "wrong_frame"
                )
            elif task_type == "task2" and not top1["answer_match"]:
                top1_failure_type = "wrong_answer"
            else:
                top1_failure_type = "correct"

        if video_rank is None:
            failure_stage = "retrieval_failure"
        elif frame_rank is None:
            failure_stage = (
                "alignment_failure" if task_type == "task3" else "localization_failure"
            )
        elif task_type == "task2" and correct_rank is None:
            failure_stage = "answer_failure"
        else:
            failure_stage = "correct_candidate_available"

        result = {
            "query_id": query_id,
            "task_type": task_type,
            "query": gt_item.get("query", ""),
            "gt_video": gt_item.get("video_id", ""),
            "gt_video_rank": video_rank,
            "video_rank_source": retrieval_source,
            "best_gt_frame_candidate_rank": frame_rank,
            "first_fully_correct_rank": correct_rank,
            "prediction_count": len(preds),
            "invalid_prediction_count": sum(
                1 for item in scored if not item["schema_valid"]
            ),
            "top1_failure_type": top1_failure_type,
            "failure_stage": failure_stage,
            "final_score": _mean(list(r_at_k.values())),
            "r_at_k": r_at_k,
            "video_recall": video_recall,
            "frame_recall": frame_recall,
        }

        if task_type == "task2":
            result["qa"] = {
                "top1_video_match": bool(top1 and top1["video_match"]),
                "top1_frame_match": bool(top1 and top1["frame_match"]),
                "top1_answer_match": bool(top1 and top1["answer_match"]),
                "classification": top1_failure_type,
            }

        if task_type == "task3":
            event_count = len(gt_item.get("events_dicts") or [])
            result["trake"] = {
                "event_count": event_count,
                "top1_event_matches": top1["event_matches"] if top1 else [False] * event_count,
                "top1_correct_event_count": top1["correct_event_count"] if top1 else 0,
                "top1_event_accuracy": top1["event_accuracy"] if top1 else 0.0,
                "top1_temporal_order_error": bool(
                    top1 and top1["temporal_order_error"]
                ),
                "best_correct_event_count": max(
                    [item["correct_event_count"] for item in scored] or [0]
                ),
                "best_event_accuracy": max(
                    [item["event_accuracy"] or 0.0 for item in scored] or [0.0]
                ),
            }

        return result

    def evaluate_query(self, query_id, task_type, predictions):
        """API tương thích với code cũ: trả về final_score và list R@k."""
        detail = self.evaluate_query_detailed(query_id, task_type, predictions)
        return detail["final_score"], [
            detail["r_at_k"]["R@%d" % k] for k in K_VALUES
        ]

    @staticmethod
    def _aggregate_details(details):
        if not details:
            empty = {
                "query_count": 0,
                "Final_Score": 0.0,
            }
            for k in K_VALUES:
                empty["R@%d" % k] = 0.0
            for k in VIDEO_K_VALUES:
                empty["Video Recall@%d" % k] = 0.0
            for k in FRAME_K_VALUES:
                empty["Frame Recall@%d" % k] = 0.0
            return empty

        output = {
            "query_count": len(details),
            "Final_Score": _mean([item["final_score"] for item in details]),
        }
        for k in K_VALUES:
            key = "R@%d" % k
            output[key] = _mean([item["r_at_k"][key] for item in details])
        for k in VIDEO_K_VALUES:
            key = "Video Recall@%d" % k
            output[key] = _mean([item["video_recall"][key] for item in details])
        for k in FRAME_K_VALUES:
            key = "Frame Recall@%d" % k
            output[key] = _mean([item["frame_recall"][key] for item in details])
        return output

    @staticmethod
    def _aggregate_qa(details):
        qa_details = [item for item in details if item["task_type"] == "task2"]
        counts = defaultdict(int)
        for item in qa_details:
            counts[item["qa"]["classification"]] += 1
        total = len(qa_details)
        return {
            "query_count": total,
            "top1_classification_counts": dict(counts),
            "top1_classification_rates": {
                key: (value / total if total else 0.0)
                for key, value in counts.items()
            },
        }

    @staticmethod
    def _aggregate_trake(details):
        trake_details = [item for item in details if item["task_type"] == "task3"]
        per_event_hits = defaultdict(list)
        for item in trake_details:
            for event_idx, match in enumerate(item["trake"]["top1_event_matches"], start=1):
                per_event_hits[event_idx].append(float(match))

        return {
            "query_count": len(trake_details),
            "event_accuracy_at_top1": {
                "event_%d" % event_idx: _mean(values)
                for event_idx, values in sorted(per_event_hits.items())
            },
            "mean_correct_events_at_top1": _mean(
                [item["trake"]["top1_correct_event_count"] for item in trake_details]
            ),
            "mean_event_accuracy_at_top1": _mean(
                [item["trake"]["top1_event_accuracy"] for item in trake_details]
            ),
            "mean_best_correct_events_in_top100": _mean(
                [item["trake"]["best_correct_event_count"] for item in trake_details]
            ),
            "temporal_order_error_count_at_top1": sum(
                int(item["trake"]["top1_temporal_order_error"])
                for item in trake_details
            ),
            "temporal_order_error_rate_at_top1": _mean(
                [
                    float(item["trake"]["top1_temporal_order_error"])
                    for item in trake_details
                ]
            ),
        }

    def evaluate_all(self, predictions_dict, retrieval_dict=None):
        predictions_dict = predictions_dict or {}
        retrieval_dict = retrieval_dict or {}
        details = []

        for task_type in ("task1", "task2", "task3"):
            task_predictions = predictions_dict.get(task_type, {})
            task_retrieval = retrieval_dict.get(task_type, {})
            for query_id in self.gt[task_type]:
                details.append(
                    self.evaluate_query_detailed(
                        query_id=query_id,
                        task_type=task_type,
                        predictions=task_predictions.get(query_id, []),
                        retrieval_candidates=task_retrieval.get(query_id),
                    )
                )

        overall = self._aggregate_details(details)
        per_task = {
            task_type: self._aggregate_details(
                [item for item in details if item["task_type"] == task_type]
            )
            for task_type in ("task1", "task2", "task3")
        }

        result = dict(overall)
        result.update(
            {
                "ground_truth": self.ground_truth_report,
                "per_task": per_task,
                "qa_diagnostics": self._aggregate_qa(details),
                "trake_diagnostics": self._aggregate_trake(details),
                "per_query": details,
            }
        )
        return result
