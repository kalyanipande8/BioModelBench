import json
import os

from src.benchmark import run_benchmark, summarize
from src.simulate_tasks import TASK_CATEGORIES


def main():
    rows = run_benchmark(n_samples=97)
    summary = summarize(rows)

    n_test_per_row = round(97 * 0.3)
    os.makedirs("results", exist_ok=True)
    output = {"n_predictions_evaluated": len(rows) * n_test_per_row, "n_task_categories": len(TASK_CATEGORIES),
              "per_model_summary": summary, "rows": rows}
    with open("results/summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps({"n_predictions_evaluated": output["n_predictions_evaluated"],
                       "n_task_categories": output["n_task_categories"],
                       "per_model_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
