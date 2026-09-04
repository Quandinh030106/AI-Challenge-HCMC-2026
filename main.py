import argparse
import json
import os
import sys
import traceback
from src.utils import FrameOCRStore

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.evaluation.debug_analyzer import DebugAnalyzer
from src.preprocessing.query_processor import QueryProcessor
from src.search.dense_search import DenseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.search.object_search import ObjectSearcher
from src.search.sparse_search import SparseSearcher
from src.search.sequence_search import rerank_sequence_aware_kis
from src.search.temporal_refiner import TemporalRefiner
from src.search.visual_reranker import VisualReRanker
from src.tasks.task1_kis import generate_diversity_top100_kis
from src.tasks.task2_vqa import (
    build_task2_top100_predictions,
    solve_task2,
)
from src.tasks.task3_trake import solve_task3_batch
from src.utils import load_config, normalize_query_item
from src.search.coarse_filter import CoarseFilter


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def find_ground_truth(explicit_path, metadata_dir):
    """
    Chỉ tự động dùng local_val_gt.json.

    Không fallback sang src/label/sample.json hoặc Task*.json vì các file đó
    là placeholder/smoke-test và có thể tạo metric gây hiểu nhầm.
    """
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                "Ground Truth không tồn tại: %s" % explicit_path
            )
        return explicit_path

    candidates = [
        os.path.join(metadata_dir, "local_val_gt.json") if metadata_dir else "",
        os.path.join("data", "metadata", "local_val_gt.json"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.path.getsize(candidate) > 5:
            return candidate
    return None


def task_items(ground_truth, task_type):
    items = ground_truth.get(task_type, [])
    if isinstance(items, dict):
        return list(items.values())
    if isinstance(items, list):
        return items
    return []


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def run_video_retrieval(
    query_text,
    task_type,
    dense_searcher,
    sparse_searcher,
    query_processor,
    object_searcher,
    coarse_filter=None,
):
    """Chạy đúng retrieval stage hiện dùng trong Codabench pipeline."""
    query_info = query_processor.process(query_text)
    intent = query_info["intent_info"]

    coarse_video_candidates = None

    if coarse_filter is not None:
        try:
            coarse_video_candidates = coarse_filter.filter(
                query_info,
                top_k=100,
            )

        except Exception as exc:
            print(
                "[WARNING] CoarseFilter failed: %s"
                % exc
            )

            coarse_video_candidates = None

    # Q&A chỉ dùng visual/context description cho retrieval.
    search_text = query_text
    dense_results = dense_searcher.search(
        query_info["semantic_views"],
        top_k_videos=100,
        candidate_video_ids=coarse_video_candidates,
    )
    sparse_results = sparse_searcher.search(
        search_text,
        top_k_videos=50,
    )

    # Snapshot gọn trước ObjectSearcher vì module đó có thể cập nhật nested
    # dense_info tại chỗ. Trace cần giữ cả điểm Dense gốc và điểm sau boost.
    dense_trace_results = [
        {
            "video_id": item.get("video_id"),
            "max_score": item.get("max_score"),
            "best_frame_idx": item.get("best_frame_idx"),
        }
        for item in dense_results
    ]

    dense_weight = 0.4 if task_type == "task2" else intent["dense_weight"]
    sparse_weight = 0.6 if task_type == "task2" else intent["sparse_weight"]

    fused = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        dense_dict=getattr(dense_searcher, "last_dense_dict", None),
    )
    pre_object_fused = [dict(item) for item in fused]
    fused = object_searcher.boost_candidates(
        fused,
        "%s %s" % (query_text, query_info.get("query_en", "")),
    )

    sequence_trace = None
    if task_type == "task1":
        fused, sequence_trace = rerank_sequence_aware_kis(
            query_text=query_text,
            fused_candidates=fused,
            dense_searcher=dense_searcher,
            sparse_searcher=sparse_searcher,
            query_processor=query_processor,
            config=dense_searcher.config,
            pre_object_candidates=pre_object_fused,
        )

    return {
        "query_info": query_info,
        "dense_results": dense_results,
        "dense_trace_results": dense_trace_results,
        "sparse_results": sparse_results,
        "fused": fused,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "sequence_trace": sequence_trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Dev-set Ground Truth thật. Nếu bỏ trống, chỉ tìm local_val_gt.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/evaluation",
        help="Thư mục lưu predictions, retrieval trace và debug report.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Dừng ngay ở lỗi pipeline đầu tiên thay vì ghi lỗi và chạy query tiếp theo.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    keyframes_dir = config["data"].get("keyframes_dir")
    metadata_dir = config["data"].get("metadata_dir")
    map_keyframes_dir = config["data"].get("map_keyframes_dir") or metadata_dir

    gt_path = find_ground_truth(args.ground_truth, metadata_dir)

    print("--- KHOI CHAY PIPELINE TIM KIEM VIDEO ---")
    dense_searcher = DenseSearcher(config)
    coarse_filter = CoarseFilter(config)


    kis_ocr_cfg = config.get("search", {}).get("kis_ocr_boost", {})
    ocr_boost_enabled = bool(kis_ocr_cfg.get("enabled", False))
    ocr_boost_weight = float(kis_ocr_cfg.get("weight", 0.0))
    apply_only_ocr_intent = bool(kis_ocr_cfg.get("apply_only_ocr_intent", True))
    ocr_store = FrameOCRStore(metadata_dir, keyframes_dir) if ocr_boost_enabled else None

    query_processor = QueryProcessor()
    temporal_refiner = TemporalRefiner(config, dense_searcher)

    if not gt_path:
        print("\nKhông tìm thấy local dev-set Ground Truth thật.")
        print("Không tự động dùng src/label/sample.json hoặc Task*.json.")
        print("Chạy smoke query để kiểm tra DenseSearcher:")
        test_query = "một diễn giả đang phát biểu trước máy quay"
        query_info = query_processor.process(test_query)
        dense_results = dense_searcher.search(
            query_info["semantic_views"],
            top_k_videos=3,
        )
        for rank, result in enumerate(dense_results, start=1):
            print(
                "Top %d: Video=%s, Score=%.4f, Best Keyframe Index=%s"
                % (
                    rank,
                    result["video_id"],
                    result["max_score"],
                    result["best_frame_idx"],
                )
            )
        print(
            "\nMuốn đánh giá Prompt 3, chạy lại với --ground-truth "
            "data/dev/local_val_gt.json"
        )
        return

    print("\nGround Truth: %s" % gt_path)
    ground_truth = load_ground_truth(gt_path)

    # Các module dưới đây khớp với pipeline tạo submission hiện tại.
    sparse_searcher = SparseSearcher(config)
    object_searcher = ObjectSearcher(config)
    vlm_model_name = config.get("models", {}).get(
        "vlm_model",
        "Qwen/Qwen2-VL-2B-Instruct",
    )
    visual_reranker = (
        VisualReRanker(vlm_model_name)
        if task_items(ground_truth, "task1")
        else None
    )

    predictions_dict = {"task1": {}, "task2": {}, "task3": {}}
    retrieval_dict = {"task1": {}, "task2": {}, "task3": {}}
    temporal_refinement_dict = {}
    qa_diagnostic_dict = {}
    pipeline_errors = []

    def record_error(task_type, query_id, exc):
        error = {
            "task_type": task_type,
            "query_id": query_id,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        pipeline_errors.append(error)
        print(
            "[ERROR] %s/%s | %s: %s"
            % (task_type, query_id, type(exc).__name__, exc)
        )
        if args.fail_fast:
            traceback.print_exc()
            raise exc

    print("\nĐang xử lý Task 1 (Textual KIS)...")
    for raw_item in task_items(ground_truth, "task1"):
        item = normalize_query_item(raw_item)
        query_id = item["query_id"]
        try:
            retrieval = run_video_retrieval(
                item["query"],
                "task1",
                dense_searcher,
                sparse_searcher,
                query_processor,
                object_searcher,
                coarse_filter,
            )
            fused = retrieval["fused"]
            if visual_reranker is not None:
                fused = visual_reranker.rerank_candidates(
                    fused,
                    item["query"],
                    keyframes_dir,
                    top_n_verify=5,
                )

            retrieval_dict["task1"][query_id] = DebugAnalyzer.build_retrieval_trace(
                retrieval["dense_trace_results"],
                retrieval["sparse_results"],
                fused,
                dense_weight=retrieval["dense_weight"],
                sparse_weight=retrieval["sparse_weight"],
            )

            intent_name = retrieval["query_info"]["intent_info"].get("intent")
            effective_ocr_weight = (
                ocr_boost_weight
                if ocr_boost_enabled and (not apply_only_ocr_intent or intent_name == "OCR_TEXT")
                else 0.0
            )
            coarse_predictions = generate_diversity_top100_kis(
                fused,
                keyframes_dir,
                metadata_dir=map_keyframes_dir,
                total_preds=100,
                ocr_store=ocr_store,
                query_text=item["query"],
                ocr_weight=effective_ocr_weight,
            )
            refined_predictions, refinement_trace = (
                temporal_refiner.refine_kis_predictions(
                    query_id=query_id,
                    query_text=item["query"],
                    prompt_ensemble=retrieval["query_info"]["prompt_ensemble"],
                    coarse_predictions=coarse_predictions,
                    fused_candidates=fused,
                    query_processor=query_processor,
                )
            )
            predictions_dict["task1"][query_id] = refined_predictions
            temporal_refinement_dict[query_id] = refinement_trace
        except Exception as exc:
            # Lỗi được ghi rõ vào report; không silent pass.
            predictions_dict["task1"].setdefault(query_id, [])
            record_error("task1", query_id, exc)

    print("\nĐang xử lý Task 2 (Visual Q&A)...")
    for raw_item in task_items(ground_truth, "task2"):
        item = normalize_query_item(raw_item)
        query_id = item["query_id"]
        try:
            retrieval = run_video_retrieval(
                item["query"],
                "task2",
                dense_searcher,
                sparse_searcher,
                query_processor,
                object_searcher,
                coarse_filter,
            )
            fused = retrieval["fused"]

            # Video Recall được đo trước khi VQA dùng answer để promote candidate.
            retrieval_dict["task2"][query_id] = DebugAnalyzer.build_retrieval_trace(
                retrieval["dense_trace_results"],
                retrieval["sparse_results"],
                fused,
                dense_weight=retrieval["dense_weight"],
                sparse_weight=retrieval["sparse_weight"],
            )

            answer_result = solve_task2(
                item["query"],
                item["question"],
                fused,
                keyframes_dir,
                model_id=vlm_model_name,
                metadata_dir=map_keyframes_dir,
                object_searcher=object_searcher,
                ocr_dir=metadata_dir,
                qa_config=config.get("search", {}).get("qa_evidence", {}),
                temporal_refiner=temporal_refiner,
                query_processor=query_processor,
                query_id=query_id,
            )
            qa_diagnostic_dict[query_id] = answer_result

            promoted_idx = int(answer_result.get("promoted_idx", 0) or 0)
            if 0 < promoted_idx < len(fused):
                promoted_candidate = fused.pop(promoted_idx)
                fused.insert(0, promoted_candidate)

            predictions_dict["task2"][query_id] = (
                build_task2_top100_predictions(
                    fused_candidates=fused,
                    answer_result=answer_result,
                    keyframes_dir=keyframes_dir,
                    metadata_dir=map_keyframes_dir,
                    total_preds=100,
                    qa_config=config.get("search", {}).get("qa_evidence", {}),
                )
            )
        except Exception as exc:
            predictions_dict["task2"].setdefault(query_id, [])
            record_error("task2", query_id, exc)

    print("\nĐang xử lý Task 3 (TRAKE)...")
    for raw_item in task_items(ground_truth, "task3"):
        item = normalize_query_item(raw_item)
        query_id = item["query_id"]
        try:
            retrieval = run_video_retrieval(
                item["query"],
                "task3",
                dense_searcher,
                sparse_searcher,
                query_processor,
                object_searcher,
                coarse_filter,
            )
            fused = retrieval["fused"]
            retrieval_dict["task3"][query_id] = DebugAnalyzer.build_retrieval_trace(
                retrieval["dense_trace_results"],
                retrieval["sparse_results"],
                fused,
                dense_weight=retrieval["dense_weight"],
                sparse_weight=retrieval["sparse_weight"],
            )
            predictions_dict["task3"][query_id] = solve_task3_batch(
                item["events"],
                fused,
                keyframes_dir,
                dense_searcher,
                metadata_dir=map_keyframes_dir,
                query_processor=query_processor,
                total_preds=100,
                config=config,
                temporal_refiner=temporal_refiner,
                query_id=query_id,
            )
        except Exception as exc:
            predictions_dict["task3"].setdefault(query_id, [])
            record_error("task3", query_id, exc)

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_path = os.path.join(args.output_dir, "predictions.json")
    retrieval_path = os.path.join(args.output_dir, "retrieval_trace.json")
    temporal_path = os.path.join(
        args.output_dir,
        "temporal_refinement_trace.json",
    )
    qa_path = os.path.join(args.output_dir, "qa_diagnostic_trace.json")
    report_path = os.path.join(args.output_dir, "debug_report.json")

    save_json(predictions_dict, predictions_path)
    save_json(retrieval_dict, retrieval_path)
    save_json(temporal_refinement_dict, temporal_path)
    save_json(qa_diagnostic_dict, qa_path)

    analyzer = DebugAnalyzer(ground_truth, ground_truth_source=gt_path)
    report = analyzer.analyze_all(
        predictions_dict=predictions_dict,
        retrieval_dict=retrieval_dict,
        pipeline_errors=pipeline_errors,
    )
    analyzer.save_json(report, report_path)
    analyzer.print_summary(report)

    print("\nĐã lưu:")
    print(" - Predictions    : %s" % os.path.abspath(predictions_path))
    print(" - Retrieval trace: %s" % os.path.abspath(retrieval_path))
    print(" - Temporal trace : %s" % os.path.abspath(temporal_path))
    print(" - Q&A trace      : %s" % os.path.abspath(qa_path))
    print(" - Debug report   : %s" % os.path.abspath(report_path))


if __name__ == "__main__":
    main()
