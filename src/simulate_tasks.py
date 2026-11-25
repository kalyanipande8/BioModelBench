"""Simulate per-task-category prediction datasets. Each of 14 biological
task categories gets its own synthetic classification problem with a known
difficulty level (controls how separable the classes are), independent of
which model will later be benchmarked on it."""
import numpy as np

TASK_CATEGORIES = [
    "variant_pathogenicity", "protein_stability", "drug_response", "gene_essentiality",
    "disease_subtype", "toxicity_prediction", "binding_affinity_class", "expression_level_class",
    "splice_site", "promoter_activity", "cell_type", "metabolic_pathway_class",
    "immunogenicity", "ppi_interaction",
]
assert len(TASK_CATEGORIES) == 14


def simulate_task_data(task_name: str, n_samples: int = 90, seed: int = 0):
    rng = np.random.default_rng(abs(hash(task_name)) % (2**32) + seed)
    difficulty = rng.uniform(0.3, 1.5)  # higher = harder (more class overlap)
    n_features = 10
    X = rng.normal(size=(n_samples, n_features))
    true_weights = rng.normal(size=n_features)
    logits = X @ true_weights / difficulty
    y = (logits > np.median(logits)).astype(int)
    return X, y, difficulty


def simulate_all_tasks(n_samples: int = 90, seed: int = 0):
    return {task: simulate_task_data(task, n_samples=n_samples, seed=seed) for task in TASK_CATEGORIES}
