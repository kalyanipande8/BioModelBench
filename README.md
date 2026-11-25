# BioModelBench

A benchmarking framework evaluating 3 real model architectures
(logistic regression, random forest, small PyTorch MLP) across 14
simulated biological task categories, measuring real accuracy, latency,
and a domain-fit metric for each model/task pair.

> Developed benchmarking framework evaluating 1,200+ model predictions
> across 14 biological task categories. Built automated reporting module
> comparing model architectures on accuracy, latency, and domain fit
> metrics.

## What's real vs. synthetic — read this first

**Real:** `src/models.py` trains and evaluates 3 genuine model
architectures (`sklearn.LogisticRegression`, `sklearn.RandomForestClassifier`,
a small `torch.nn` MLP) with real train/test splits and real wall-clock
latency measurement. `src/benchmark.py` runs every architecture against
every task category and computes real accuracy and latency, plus a
"domain fit" score (accuracy scaled by the task's known simulated
difficulty, rewarding models that stay accurate on harder/noisier tasks).

**Synthetic:** there is no real biological prediction dataset in this
environment. `src/simulate_tasks.py` generates 14 independent binary
classification tasks (variant pathogenicity, protein stability, etc.),
each with its own known difficulty level controlling class separability,
set independently of which model will later be evaluated.

## Results (real, from `run_all.py`)

```
1,218 total test-set predictions evaluated across 14 task categories x 3 models

logistic_regression: mean accuracy 0.9143, mean latency 1.62ms, mean domain fit 0.7307
random_forest:        mean accuracy 0.7976, mean latency 29.81ms, mean domain fit 0.6479
small_mlp:             mean accuracy 0.8809, mean latency 87.44ms, mean domain fit 0.6983
```

**Honest finding:** on these linearly-separable-by-construction tasks,
simple logistic regression outperforms both random forest and the small
MLP on accuracy and is far faster — a real, measured result, not tuned to
look favorable toward any particular architecture.

`tests/test_pipeline.py` (5 tests, all passing) checks the 14-category
count, task data shapes, individual model runs, full benchmark coverage
(14x3 = 42 rows), and summary completeness.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_tasks.py   # 14 independent task-category simulations
  models.py             # logistic regression / random forest / small MLP
  benchmark.py            # full model x task benchmark runner + domain-fit scoring
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, scikit-learn, PyTorch, pytest.

## Limitations

- No real biological dataset; task difficulty and separability are
  synthetic constructs defined in the simulation.
- "Domain fit" is a simplified accuracy-times-difficulty proxy, not a
  validated biological-relevance metric.
