from collections import Counter, defaultdict


def summarize_failures(query_results):
    """Tổng hợp failure stage và lỗi Top-1 theo task."""
    failure_stage = Counter()
    top1_failure = Counter()
    per_task = defaultdict(
        lambda: {
            "total_queries": 0,
            "failure_stage": Counter(),
            "top1_failure_type": Counter(),
        }
    )

    for item in query_results:
        stage = item.get("failure_stage", "unknown")
        top1 = item.get("top1_failure_type", "unknown")
        task_type = item.get("task_type", "unknown")

        failure_stage[stage] += 1
        top1_failure[top1] += 1
        per_task[task_type]["total_queries"] += 1
        per_task[task_type]["failure_stage"][stage] += 1
        per_task[task_type]["top1_failure_type"][top1] += 1

    total = len(query_results)

    def rates(counter, denominator):
        return {
            key: (value / denominator if denominator else 0.0)
            for key, value in counter.items()
        }

    per_task_output = {}
    for task_type, values in per_task.items():
        task_total = values["total_queries"]
        per_task_output[task_type] = {
            "total_queries": task_total,
            "failure_stage_counts": dict(values["failure_stage"]),
            "failure_stage_rates": rates(values["failure_stage"], task_total),
            "top1_failure_counts": dict(values["top1_failure_type"]),
            "top1_failure_rates": rates(values["top1_failure_type"], task_total),
        }

    return {
        "total_queries": total,
        "failure_stage_counts": dict(failure_stage),
        "failure_stage_rates": rates(failure_stage, total),
        "top1_failure_counts": dict(top1_failure),
        "top1_failure_rates": rates(top1_failure, total),
        "per_task": per_task_output,
    }
