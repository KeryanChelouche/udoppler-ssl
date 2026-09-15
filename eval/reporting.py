"""Metrics reporting: console tables, JSON persistence, and plots."""
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print a single evaluation result to stdout."""
    print(f"\nModel:   {results['model']}")
    print(
        f"Dataset: {results['dataset']}  "
        f"({results['n_samples']} samples, {results['n_classes']} classes)"
    )
    print("-" * 72)
    print(f"  {'probe':<20}  {'accuracy':>20}   {'macro-F1':>20}")
    print("-" * 72)
    for probe_name, probe_res in results["probes"].items():
        acc = f"{probe_res['mean_acc'] * 100:.2f}% ± {probe_res['std_acc'] * 100:.2f}%"
        f1  = f"{probe_res['mean_f1']  * 100:.2f}% ± {probe_res['std_f1']  * 100:.2f}%"
        print(f"  {probe_name:<20}  {acc:>20}   {f1:>20}")
    print()


def save_results(results: Dict[str, Any], path: Path) -> None:
    """Write results dict as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def make_comparison_table(
    results_dir: Path,
    datasets: List[str] | None = None,
    models: List[str] | None = None,
) -> str:
    """Read all JSON result files and return a Markdown comparison table.

    Rows are (dataset, model) pairs sorted so results for the same dataset
    are adjacent; columns are probe names.  Values are "mean% ± std%".

    Parameters
    ----------
    results_dir:
        Directory containing .json result files.
    datasets:
        If provided, only include rows whose dataset name is in this list.
    models:
        If provided, only include rows whose model name is in this list.
    """
    rows: List[Dict] = []
    for p in sorted(results_dir.glob("*.json")):
        data = json.loads(p.read_text())
        if datasets and data["dataset"] not in datasets:
            continue
        if models and data["model"] not in models:
            continue
        row: Dict[str, Any] = {
            "model": data["model"],
            "dataset": data["dataset"],
        }
        for probe_name, probe_res in data["probes"].items():
            row[f"{probe_name}_acc"] = (
                f"{probe_res['mean_acc'] * 100:.2f} ± {probe_res['std_acc'] * 100:.2f}"
            )
            row[f"{probe_name}_f1"] = (
                f"{probe_res['mean_f1'] * 100:.2f} ± {probe_res['std_f1'] * 100:.2f}"
            )
        rows.append(row)

    if not rows:
        return "No results found."

    # Deduplicate: keep only the latest run per (model, dataset) pair
    seen: Dict[tuple, Dict] = {}
    for row in rows:
        key = (row["model"], row["dataset"])
        seen[key] = row   # later files overwrite earlier ones (sorted by name = timestamp)

    # Sort by (dataset, model) so same-dataset rows are adjacent
    rows = sorted(seen.values(), key=lambda r: (r["dataset"], r["model"]))

    all_probes = sorted({k for r in rows for k in r if k not in ("model", "dataset")})
    headers = ["dataset", "model"] + all_probes
    col_w = [
        max(len(h), max(len(str(r.get(h, "-"))) for r in rows)) + 2
        for h in headers
    ]

    def fmt_row(vals: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    lines = [fmt_row(headers), sep] + [fmt_row([r.get(h, "-") for h in headers]) for r in rows]
    return "\n".join(lines)


def make_cross_comparison_table(
    results_dir: Path,
    models: List[str] | None = None,
    datasets: List[str] | None = None,
) -> str:
    """Read cross-eval JSON files and return a Markdown comparison table.

    Rows are (train→test, model) pairs; columns are probe metrics.
    Values are single percentages (no folds / std).

    Parameters
    ----------
    datasets:
        If provided, only include rows where either the train or test
        dataset is in this list.
    """
    rows: List[Dict] = []
    for p in sorted(results_dir.glob("*.json")):
        data = json.loads(p.read_text())
        # Skip non-cross-eval files (they have "dataset" instead of "train_dataset").
        if "train_dataset" not in data:
            continue
        if models and data["model"] not in models:
            continue
        if datasets and not (
            data["train_dataset"] in datasets or data["test_dataset"] in datasets
        ):
            continue
        direction = f"{data['train_dataset']} → {data['test_dataset']}"
        row: Dict[str, Any] = {
            "model": data["model"],
            "direction": direction,
        }
        for probe_name, probe_res in data["probes"].items():
            row[f"{probe_name}_acc"] = f"{probe_res['acc'] * 100:.2f}"
            row[f"{probe_name}_f1"] = f"{probe_res['f1'] * 100:.2f}"
        rows.append(row)

    if not rows:
        return "No cross-eval results found."

    # Deduplicate: keep only the latest run per (model, direction) pair
    seen: Dict[tuple, Dict] = {}
    for row in rows:
        key = (row["model"], row["direction"])
        seen[key] = row
    rows = sorted(seen.values(), key=lambda r: (r["direction"], r["model"]))

    all_probes = sorted(
        {k for r in rows for k in r if k not in ("model", "direction")}
    )
    headers = ["direction", "model"] + all_probes
    col_w = [
        max(len(h), max(len(str(r.get(h, "-"))) for r in rows)) + 2
        for h in headers
    ]

    def fmt_row(vals: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    lines = [fmt_row(headers), sep] + [
        fmt_row([r.get(h, "-") for h in headers]) for r in rows
    ]
    return "\n".join(lines)


def make_few_shot_table(
    input_dirs: List[Path],
    probe: str = "knn_k10",
    metric: str = "acc",
    datasets: List[str] | None = None,
    models: List[str] | None = None,
) -> str:
    """Read few-shot JSON files and return a Markdown table.

    Rows are models, columns are shot levels.  Values are "mean ± std".

    Parameters
    ----------
    input_dirs:
        Directories containing few-shot .json result files.
    probe:
        Which probe to show (e.g. "knn_k10", "linear_C1.0").
    metric:
        "acc" or "f1".
    datasets:
        If provided, only include rows whose dataset name is in this list.
    models:
        If provided, only include rows whose model name is in this list.
    """
    # Collect latest result per (model, dataset).
    latest: Dict[tuple, Dict] = {}
    for d in input_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("run_"):
                continue
            data = json.loads(p.read_text())
            if "n_repeats" not in data:
                continue
            if datasets and data["dataset"] not in datasets:
                continue
            if models and data["model"] not in models:
                continue
            key = (data["model"], data["dataset"])
            latest[key] = data

    if not latest:
        return "No few-shot results found."

    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"

    # Gather all shot levels across all results.
    all_shots: set = set()
    for data in latest.values():
        probe_data = data["probes"].get(probe, [])
        for entry in probe_data:
            all_shots.add(entry["n_shots"])
    shot_levels = sorted(all_shots, key=lambda s: (s is None, s or 0))

    # Build rows.
    rows: List[Dict[str, str]] = []
    for (model, dataset), data in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        probe_data = data["probes"].get(probe, [])
        lookup = {e["n_shots"]: e for e in probe_data}
        row: Dict[str, str] = {"model": model, "dataset": dataset}
        for s in shot_levels:
            col = "full" if s is None else f"{s}-shot"
            if s in lookup:
                m = lookup[s][mean_key] * 100
                sd = lookup[s][std_key] * 100
                row[col] = f"{m:.1f} ± {sd:.1f}"
            else:
                row[col] = "-"
        rows.append(row)

    if not rows:
        return "No matching few-shot results."

    shot_cols = ["full" if s is None else f"{s}-shot" for s in shot_levels]
    headers = ["dataset", "model"] + shot_cols
    col_w = [
        max(len(h), max((len(str(r.get(h, "-"))) for r in rows), default=0)) + 2
        for h in headers
    ]

    def fmt_row(vals: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    lines = [fmt_row(headers), sep] + [
        fmt_row([r.get(h, "-") for h in headers]) for r in rows
    ]
    return "\n".join(lines)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
    labels: List[int] | None = None,
) -> None:
    cm_kwargs = {"normalize": "true"}
    if labels is not None:
        cm_kwargs["labels"] = labels
    cm = confusion_matrix(y_true, y_pred, **cm_kwargs)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cmap="Blues",
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
