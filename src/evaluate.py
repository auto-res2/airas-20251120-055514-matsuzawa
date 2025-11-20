import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

sns.set_style("whitegrid")

# -------------------------------------------------------------------------
#                               I/O UTILS
# -------------------------------------------------------------------------

def _json_default(obj: Any):
    """Convert numpy / pandas objects to plain Python types for JSON."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)

# -------------------------------------------------------------------------
#                        PLOTTING – PER-RUN FIGURES
# -------------------------------------------------------------------------

def _plot_learning_curve(hist: pd.DataFrame, run_id: str, outdir: Path):
    plt.figure(figsize=(7, 4))
    if "train_loss" in hist.columns:
        sns.lineplot(data=hist, x="step", y="train_loss", label="train_loss")
    if "val_accuracy" in hist.columns:
        sns.lineplot(data=hist, x="step", y="val_accuracy", label="val_accuracy")
    plt.legend()
    plt.title(f"{run_id} learning curves")
    plt.tight_layout()
    p = outdir / f"{run_id}_learning_curve.pdf"
    plt.savefig(p)
    plt.close()
    print(p)


def _plot_co2(hist: pd.DataFrame, run_id: str, outdir: Path):
    if "kgCO2" not in hist.columns:
        return
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=hist, x="step", y="kgCO2")
    plt.title(f"{run_id} cumulative CO₂")
    plt.tight_layout()
    p = outdir / f"{run_id}_co2_curve.pdf"
    plt.savefig(p)
    plt.close()
    print(p)


def _plot_confusion(summary: Dict, run_id: str, outdir: Path):
    keys = ["conf_tp", "conf_fp", "conf_fn", "conf_tn"]
    if not all(k in summary for k in keys):
        return
    cm = [
        [summary["conf_tp"], summary["conf_fp"]],
        [summary["conf_fn"], summary["conf_tn"]],
    ]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Gold-Wrong", "Gold-Right"],
        yticklabels=["Pred-Wrong", "Pred-Right"],
    )
    plt.title(f"{run_id} confusion matrix")
    plt.tight_layout()
    p = outdir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(p)
    plt.close()
    print(p)

# -------------------------------------------------------------------------
#                       PER-RUN PROCESSING
# -------------------------------------------------------------------------

def _process_run(run: "wandb.apis.public.Run", root: Path):
    run_id = run.id
    outdir = root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    hist = run.history()
    summ = run.summary._json_dict
    cfg = dict(run.config)

    _save_json(
        {
            "summary": summ,
            "config": cfg,
            "history": hist.to_dict(orient="list"),
        },
        outdir / "metrics.json",
    )

    _plot_learning_curve(hist, run_id, outdir)
    _plot_co2(hist, run_id, outdir)
    _plot_confusion(summ, run_id, outdir)

    return {
        "run_id": run_id,
        "val_accuracy": float(summ.get("final_val_accuracy", np.nan)),
        "kgCO2": float(summ.get("final_kgCO2", np.nan)),
        "wall_time_h": float(summ.get("total_wall_time_h", np.nan)),
        "method": "proposed"
        if "proposed" in run_id
        else ("baseline" if any(k in run_id for k in ["baseline", "comparative"]) else "other"),
    }

# -------------------------------------------------------------------------
#                AGGREGATION, SIGNIFICANCE TESTS & COMPARISON FIGURES
# -------------------------------------------------------------------------

def _cohens_d(a: List[float], b: List[float]):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_sd == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_sd


def _compute_stats(prop: List[float], base: List[float]):
    if len(prop) >= 2 and len(base) >= 2:
        stat_res = stats.ttest_ind(prop, base, equal_var=False)
        p = float(stat_res.pvalue)
        d = _cohens_d(prop, base)
    else:
        p, d = None, None
    return p, d


def _bar(metric_map: Dict[str, float], name: str, out: Path):
    clean = {k: v for k, v in metric_map.items() if not np.isnan(v)}
    if not clean:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(clean.keys()), y=list(clean.values()), palette="viridis")
    for i, (k, v) in enumerate(clean.items()):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.ylabel(name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p = out / f"comparison_{name.replace(' ', '_')}.pdf"
    plt.savefig(p)
    plt.close()
    print(p)


def _scatter(x_map: Dict[str, float], y_map: Dict[str, float], out: Path):
    # Require same keys in both maps
    keys = [k for k in x_map if k in y_map and not (np.isnan(x_map[k]) or np.isnan(y_map[k]))]
    if not keys:
        return
    plt.figure(figsize=(6, 4))
    for k in keys:
        plt.scatter(x_map[k], y_map[k])
        plt.text(x_map[k], y_map[k], k, fontsize=7)
    plt.xlabel("kg CO₂")
    plt.ylabel("Val acc")
    plt.tight_layout()
    p = out / "comparison_accuracy_vs_co2.pdf"
    plt.savefig(p)
    plt.close()
    print(p)


def _aggregate(per_run: List[Dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)

    acc = {r["run_id"]: r["val_accuracy"] for r in per_run}
    co2 = {r["run_id"]: r["kgCO2"] for r in per_run}
    wall = {r["run_id"]: r["wall_time_h"] for r in per_run}

    proposed_runs = [r for r in per_run if r["method"] == "proposed"]
    baseline_runs = [r for r in per_run if r["method"] == "baseline"]
    if not proposed_runs or not baseline_runs:
        raise ValueError("Need both proposed and baseline/comparative runs for aggregation.")

    best_prop = max(proposed_runs, key=lambda x: x["val_accuracy"])
    best_base = max(baseline_runs, key=lambda x: x["val_accuracy"])
    gap = (
        (best_prop["val_accuracy"] - best_base["val_accuracy"]) / best_base["val_accuracy"] * 100
    )

    # ---------------- Statistical significance tests -------------------
    stat_tests = {}
    for metric_name, metric_map in {
        "val_accuracy": acc,
        "kgCO2": co2,
        "wall_time_h": wall,
    }.items():
        prop_vals = [v for k, v in metric_map.items() if "proposed" in k]
        base_vals = [v for k, v in metric_map.items() if any(b in k for b in ["baseline", "comparative"])]
        p_val, d_val = _compute_stats(prop_vals, base_vals)
        stat_tests[metric_name] = {
            "proposed_mean": float(np.mean(prop_vals)) if prop_vals else None,
            "baseline_mean": float(np.mean(base_vals)) if base_vals else None,
            "p_value": p_val,
            "effect_size_d": d_val,
            "test": "Welch t-test" if p_val is not None else "insufficient_samples",
        }

    aggregated = {
        "primary_metric": "(1) GSM8K validation accuracy; (2) kg CO₂ emitted; (3) total wall-clock time.",
        "metrics": {"val_accuracy": acc, "kgCO2": co2, "wall_time_h": wall},
        "best_proposed": {"run_id": best_prop["run_id"], "value": best_prop["val_accuracy"]},
        "best_baseline": {"run_id": best_base["run_id"], "value": best_base["val_accuracy"]},
        "gap": gap,
        "statistical_tests": stat_tests,
    }

    _save_json(aggregated, out / "aggregated_metrics.json")

    # ---------------- Figures ------------------------------------------
    _bar(acc, "Validation accuracy", out)
    _bar(co2, "kg CO₂", out)
    _bar(wall, "Wall time (h)", out)
    _scatter(co2, acc, out)
    print(out / "aggregated_metrics.json")

# -------------------------------------------------------------------------
#                                   CLI
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("run_ids", type=str, help="JSON list of WandB run ids")
    args = p.parse_args()

    out_root = Path(args.results_dir).expanduser().resolve()
    run_ids = json.loads(args.run_ids)

    cfg = OmegaConf.load("config/config.yaml")

    api = wandb.Api()
    per_run = []
    for rid in run_ids:
        run = api.run(f"{cfg.wandb.entity}/{cfg.wandb.project}/{rid}")
        per_run.append(_process_run(run, out_root))

    _aggregate(per_run, out_root / "comparison")
    print("Evaluation pipeline completed.")

if __name__ == "__main__":
    main()