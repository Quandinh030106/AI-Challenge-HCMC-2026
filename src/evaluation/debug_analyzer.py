import json
from pathlib import Path

from src.evaluation.debug_metrics import summarize_failures
from src.evaluation.evaluator import Evaluator


def _safe_number(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _compact_prediction(prediction):
    if not isinstance(prediction, dict):
        return {"raw": str(prediction)}
    allowed = (
        "video_id",
        "frame_id",
        "frame_ids",
        "answer",
        "score",
        "quality_score",
        "promoted_idx",
        "confidence",
        "evidence_source",
        "evidence_score",
    )
    return {key: prediction.get(key) for key in allowed if key in prediction}


class DebugAnalyzer:
    """Tạo diagnostic JSON; không thay đổi prediction hay điểm số."""

    def __init__(self, ground_truth, ground_truth_source=None):
        self.evaluator = Evaluator(
            ground_truth,
            ground_truth_source=ground_truth_source,
        )

    @staticmethod
    def build_retrieval_trace(
        dense_results,
        sparse_results,
        fused_results,
        dense_weight=1.0,
        sparse_weight=1.0,
        rrf_k=60,
        limit=100,
    ):
        """
        Chuyển output Dense/Sparse/Fusion thành trace JSON gọn.

        Không ghi all_scores vì mảng này rất lớn và không JSON-serializable.
        """
        dense_lookup = {}
        for rank, item in enumerate(dense_results or [], start=1):
            video_id = str(item.get("video_id", ""))
            if video_id:
                dense_lookup[video_id] = (rank, item)

        sparse_lookup = {}
        for rank, item in enumerate(sparse_results or [], start=1):
            video_id = str(item.get("video_id", ""))
            if video_id:
                sparse_lookup[video_id] = (rank, item)

        trace = []
        for final_rank, candidate in enumerate((fused_results or [])[:limit], start=1):
            video_id = str(candidate.get("video_id", ""))
            dense_rank, dense_item = dense_lookup.get(video_id, (None, {}))
            sparse_rank, sparse_item = sparse_lookup.get(video_id, (None, {}))

            base_rrf = 0.0
            if dense_rank is not None:
                base_rrf += dense_weight * (1.0 / (rrf_k + dense_rank))
            if sparse_rank is not None:
                base_rrf += sparse_weight * (1.0 / (rrf_k + sparse_rank))

            # Sequence reranking ghi final score vao rrf_score de VisualReRanker
            # tiep tuc hoat dong. pre_sequence_rrf_score moi la diem sau Object.
            post_object_rrf = _safe_number(
                candidate.get(
                    "pre_sequence_rrf_score",
                    candidate.get("rrf_score"),
                )
            )
            dense_info = candidate.get("dense_info") or dense_item or {}

            score_components = {
                "dense_rank": dense_rank,
                "dense_score": _safe_number(dense_item.get("max_score")),
                "dense_best_keyframe_idx": dense_item.get("best_frame_idx"),
                "sparse_rank": sparse_rank,
                "sparse_score": _safe_number(sparse_item.get("sparse_score")),
                "base_rrf_score": float(base_rrf),
                "post_object_rrf_score": post_object_rrf,
                "object_or_other_delta": (
                    float(post_object_rrf - base_rrf)
                    if post_object_rrf is not None else None
                ),
                "sequence_score": _safe_number(candidate.get("sequence_score")),
                "sequence_score_components": candidate.get(
                    "sequence_score_components"
                ),
                "vlm_score": _safe_number(candidate.get("vlm_score")),
                "boosted_score": _safe_number(candidate.get("boosted_score")),
                "final_dense_score": _safe_number(dense_info.get("max_score")),
                "final_best_keyframe_idx": dense_info.get("best_frame_idx"),
            }

            trace.append(
                {
                    "video_id": video_id,
                    "final_video_rank": final_rank,
                    "score_components": score_components,
                    "sequence_event_evidence": candidate.get(
                        "sequence_event_evidence",
                        [],
                    ),
                }
            )
        return trace

    @staticmethod
    def _top_candidates_beating_gt(retrieval_trace, gt_rank, limit=10):
        if gt_rank is None:
            candidates = retrieval_trace[:limit]
        else:
            candidates = retrieval_trace[: max(0, gt_rank - 1)][:limit]
        return candidates

    def analyze_all(
        self,
        predictions_dict,
        retrieval_dict=None,
        pipeline_errors=None,
        top_prediction_limit=10,
    ):
        retrieval_dict = retrieval_dict or {}
        metrics = self.evaluator.evaluate_all(
            predictions_dict,
            retrieval_dict=retrieval_dict,
        )

        query_results = []
        for detail in metrics["per_query"]:
            task_type = detail["task_type"]
            query_id = detail["query_id"]
            retrieval_trace = retrieval_dict.get(task_type, {}).get(query_id, [])
            predictions = predictions_dict.get(task_type, {}).get(query_id, [])

            enriched = dict(detail)
            enriched["top_candidates_beating_gt"] = self._top_candidates_beating_gt(
                retrieval_trace,
                detail.get("gt_video_rank"),
            )
            enriched["top_retrieval_candidates"] = retrieval_trace[:10]
            enriched["top_submission_predictions"] = [
                _compact_prediction(pred)
                for pred in predictions[:top_prediction_limit]
            ]
            query_results.append(enriched)

        public_metrics = dict(metrics)
        public_metrics.pop("per_query", None)

        pipeline_error_list = list(pipeline_errors or [])
        ground_truth_report = self.evaluator.ground_truth_report

        benchmark_claim_allowed = (
            not ground_truth_report["is_placeholder"]
            and ground_truth_report["is_usable"]
            and not pipeline_error_list
        )

        return {
            "ground_truth": ground_truth_report,
            "benchmark_claim_allowed": benchmark_claim_allowed,
            "metrics": public_metrics,
            "failure_summary": summarize_failures(query_results),
            "pipeline_errors": pipeline_error_list,
            "queries": query_results,
        }

    @staticmethod
    def save_json(data, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)

    @staticmethod
    def print_summary(report):
        metrics = report["metrics"]
        gt_report = report["ground_truth"]

        print("\n" + "=" * 68)
        print("KET QUA EVALUATION / RETRIEVAL DIAGNOSTIC")
        print("=" * 68)

        if gt_report.get("is_placeholder"):
            print("CANH BAO: Ground Truth la placeholder/smoke-test.")
            print("Cac metric duoi day chi dung de kiem tra code, KHONG phai benchmark.")
            for reason in gt_report.get("placeholder_reasons", []):
                print(" - " + reason)

        for key in (
            "Video Recall@1",
            "Video Recall@5",
            "Video Recall@20",
            "Frame Recall@1",
            "Frame Recall@5",
            "Frame Recall@20",
            "R@1",
            "R@5",
            "R@20",
            "R@50",
            "R@100",
            "Final_Score",
        ):
            print(" %-18s: %.4f" % (key, metrics.get(key, 0.0)))

        print("=" * 68)
        for item in report["queries"]:
            print("\nQuery %s (%s)" % (item["query_id"], item["task_type"]))
            print(" - GT video rank              : %s" % item["gt_video_rank"])
            print(
                " - Best GT-frame candidate rank: %s"
                % item["best_gt_frame_candidate_rank"]
            )
            print(" - Failure stage              : %s" % item["failure_stage"])
            if item["top_candidates_beating_gt"]:
                print(" - Top candidates dang xep tren GT:")
                for candidate in item["top_candidates_beating_gt"][:5]:
                    scores = candidate["score_components"]
                    print(
                        "   #%s %s | dense=%s sparse=%s rrf=%s vlm=%s"
                        % (
                            candidate["final_video_rank"],
                            candidate["video_id"],
                            scores.get("dense_score"),
                            scores.get("sparse_score"),
                            scores.get("post_object_rrf_score"),
                            scores.get("vlm_score"),
                        )
                    )
