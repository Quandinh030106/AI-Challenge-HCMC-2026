import argparse
import json
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.evaluation.debug_analyzer import DebugAnalyzer


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def main():
    parser = argparse.ArgumentParser(
        description="Tinh metric va tao diagnostic tu prediction da luu."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--retrieval-trace",
        required=True,
        help=(
            "File retrieval_trace.json. Bắt buộc để Video Recall được tính "
            "trên video retrieval thay vì suy ra từ các dòng submission."
        ),
    )
    parser.add_argument(
        "--output",
        default="output/evaluation/debug_report.json",
    )
    args = parser.parse_args()

    ground_truth = load_json(args.ground_truth)
    predictions = load_json(args.predictions)
    retrieval_trace = load_json(args.retrieval_trace) 
    

    analyzer = DebugAnalyzer(
        ground_truth,
        ground_truth_source=args.ground_truth,
    )
    report = analyzer.analyze_all(
        predictions_dict=predictions,
        retrieval_dict=retrieval_trace,
    )
    analyzer.save_json(report, args.output)
    analyzer.print_summary(report)
    print("\nDa luu diagnostic: %s" % os.path.abspath(args.output))


if __name__ == "__main__":
    main()
