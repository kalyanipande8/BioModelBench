"""Benchmark runner: evaluate each architecture on each task category,
computing a real "domain fit" score (accuracy relative to task difficulty
as a proxy for how well a model copes with harder/noisier biological
signal, since difficulty is known from the simulation's ground truth)."""
from .simulate_tasks import simulate_all_tasks, TASK_CATEGORIES
from .models import MODEL_NAMES, run_architecture


def run_benchmark(n_samples: int = 90, seed: int = 0):
    tasks = simulate_all_tasks(n_samples=n_samples, seed=seed)
    rows = []
    for task_name in TASK_CATEGORIES:
        X, y, difficulty = tasks[task_name]
        for model_name in MODEL_NAMES:
            result = run_architecture(model_name, X, y)
            domain_fit = result["accuracy"] * difficulty  # rewards robustness on harder tasks
            rows.append({
                "task": task_name, "model": model_name, "difficulty": round(difficulty, 3),
                "accuracy": round(result["accuracy"], 4), "latency_ms": round(result["latency_ms"], 3),
                "domain_fit": round(domain_fit, 4),
            })
    return rows


def summarize(rows):
    summary = {}
    for model_name in MODEL_NAMES:
        model_rows = [r for r in rows if r["model"] == model_name]
        summary[model_name] = {
            "mean_accuracy": round(sum(r["accuracy"] for r in model_rows) / len(model_rows), 4),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in model_rows) / len(model_rows), 4),
            "mean_domain_fit": round(sum(r["domain_fit"] for r in model_rows) / len(model_rows), 4),
        }
    return summary
