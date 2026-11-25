import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_tasks import simulate_all_tasks, TASK_CATEGORIES, simulate_task_data
from src.models import MODEL_NAMES, run_architecture
from src.benchmark import run_benchmark, summarize


def test_14_task_categories():
    assert len(TASK_CATEGORIES) == 14


def test_task_data_shapes():
    X, y, difficulty = simulate_task_data("variant_pathogenicity", n_samples=50)
    assert X.shape[0] == 50
    assert set(y.tolist()).issubset({0, 1})
    assert difficulty > 0


def test_model_runs_and_returns_metrics():
    X, y, _ = simulate_task_data("protein_stability", n_samples=60)
    result = run_architecture("logistic_regression", X, y)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["latency_ms"] >= 0


def test_benchmark_covers_all_model_task_pairs():
    rows = run_benchmark(n_samples=40)
    assert len(rows) == len(TASK_CATEGORIES) * len(MODEL_NAMES)


def test_summary_has_all_models():
    rows = run_benchmark(n_samples=40)
    summary = summarize(rows)
    assert set(summary.keys()) == set(MODEL_NAMES)
