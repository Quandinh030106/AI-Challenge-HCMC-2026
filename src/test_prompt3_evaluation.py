import json
import os
import sys
import tempfile


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.evaluation.debug_analyzer import DebugAnalyzer
from src.evaluation.evaluator import Evaluator


def build_fixture():
    ground_truth = {
        "_meta": {"status": "smoke_test"},
        "task1": [
            {
                "query_id": "kis_1",
                "query": "KIS unit test",
                "video_id": "V_GT_1",
                "frame_start": 100,
                "frame_end": 110,
            }
        ],
        "task2": [
            {
                "query_id": "qa_1",
                "query": "QA unit test",
                "question": "Màu gì?",
                "video_id": "V_GT_2",
                "frame_start": 200,
                "frame_end": 210,
                "answer": "màu xanh",
                "acceptable_answers": ["xanh", "blue"],
            }
        ],
        "task3": [
            {
                "query_id": "trake_1",
                "query": "TRAKE unit test",
                "video_id": "V_GT_3",
                "events": [
                    {"name": "E1", "frame_start": 10, "frame_end": 20},
                    {"name": "E2", "frame_start": 30, "frame_end": 40},
                    {"name": "E3", "frame_start": 50, "frame_end": 60},
                ],
            }
        ],
    }

    predictions = {
        "task1": {
            "kis_1": [
                {"video_id": "V_WRONG", "frame_id": 105},
                {"video_id": "V_GT_1", "frame_id": 999},
                {"video_id": "V_GT_1", "frame_id": 105},
            ]
        },
        "task2": {
            "qa_1": [
                {"video_id": "V_GT_2", "frame_id": 205, "answer": "màu đỏ"},
                {"video_id": "V_GT_2", "frame_id": 205, "answer": "blue"},
            ]
        },
        "task3": {
            "trake_1": [
                {"video_id": "V_GT_3", "frame_ids": [15, 55, 35]},
                {"video_id": "V_GT_3", "frame_ids": [15, 35, 55]},
            ]
        },
    }

    retrieval = {
        "task1": {
            "kis_1": [
                {"video_id": "V_WRONG"},
                {"video_id": "V_GT_1"},
            ]
        },
        "task2": {"qa_1": [{"video_id": "V_GT_2"}]},
        "task3": {"trake_1": [{"video_id": "V_GT_3"}]},
    }
    return ground_truth, predictions, retrieval


def main():
    ground_truth, predictions, retrieval = build_fixture()
    evaluator = Evaluator(ground_truth, ground_truth_source="smoke_test.json")

    assert not evaluator.check_answer_match("màu đỏ", evaluator.gt["task2"]["qa_1"])
    assert evaluator.check_answer_match("blue", evaluator.gt["task2"]["qa_1"])

    metrics = evaluator.evaluate_all(predictions, retrieval_dict=retrieval)

    assert abs(metrics["Video Recall@1"] - (2.0 / 3.0)) < 1e-9
    assert abs(metrics["Frame Recall@1"] - (1.0 / 3.0)) < 1e-9
    assert abs(metrics["R@1"] - (1.0 / 9.0)) < 1e-9
    assert metrics["R@5"] == 1.0

    per_query = {item["query_id"]: item for item in metrics["per_query"]}
    assert per_query["kis_1"]["gt_video_rank"] == 2
    assert per_query["kis_1"]["best_gt_frame_candidate_rank"] == 3
    assert per_query["qa_1"]["qa"]["classification"] == "wrong_answer"
    assert per_query["trake_1"]["trake"]["top1_temporal_order_error"]
    assert per_query["trake_1"]["trake"]["top1_correct_event_count"] == 1

    analyzer = DebugAnalyzer(ground_truth, ground_truth_source="smoke_test.json")
    report = analyzer.analyze_all(predictions, retrieval_dict=retrieval)
    assert report["benchmark_claim_allowed"] is False

    with tempfile.TemporaryDirectory() as temp_dir:
        output = os.path.join(temp_dir, "debug_report.json")
        analyzer.save_json(report, output)
        with open(output, "r", encoding="utf-8") as file_obj:
            saved = json.load(file_obj)
        assert saved["queries"][0]["query_id"] == "kis_1"

    print("PROMPT 3 EVALUATOR TESTS: ALL PASSED")


if __name__ == "__main__":
    main()
