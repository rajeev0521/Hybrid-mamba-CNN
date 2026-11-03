#!/usr/bin/env python3
"""
Comprehensive model comparison script.
Trains CNN, ViM, and Hybrid Mamba-CNN models and generates comparison diagrams.
"""

import subprocess
import sys
import json
import time
import os
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Setup ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / ".." / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
LOG_DIR = RESULTS_DIR / "logs"

for d in [RESULTS_DIR, FIGURES_DIR, SUMMARIES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training log file
LOG_FILE = LOG_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── Model Configs ─────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "CNN Baseline",
        "config": "configs/cnn_config.yaml",
        "key":    "cnn_baseline_isic2019",
        "color":  "#4E79A7",
        "marker": "o",
    },
    {
        "name": "Vision Mamba",
        "config": "configs/vim_config.yaml",
        "key":    "vim_baseline_isic2019",
        "color":  "#F28E2B",
        "marker": "s",
    },
    {
        "name": "Hybrid Mamba-CNN",
        "config": "configs/hybrid_config.yaml",
        "key":    "hybrid_mamba_cnn_isic2019",
        "color":  "#59A14F",
        "marker": "^",
    },
]


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(model_info: dict, extra_args: list = None) -> dict:
    """Run training for one model, streaming output to the log."""
    name   = model_info["name"]
    config = model_info["config"]

    log.info("=" * 70)
    log.info(f"Training: {name}")
    log.info(f"Config  : {config}")
    log.info("=" * 70)

    cmd = [
        sys.executable, str(PROJECT_DIR / "train.py"),
        "--config", str(PROJECT_DIR / config),
    ]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        log.info(f"[{name}] {line}")

    proc.wait()
    elapsed = time.time() - start

    if proc.returncode != 0:
        log.error(f"Training FAILED for {name} (exit code {proc.returncode})")
        return {"status": "failed", "training_time": elapsed}

    log.info(f"Finished {name} in {elapsed:.1f}s")
    return {"status": "ok", "training_time": elapsed}


def load_summary(key: str) -> dict:
    """Load saved JSON summary for a model."""
    path = SUMMARIES_DIR / f"{key}_summary.json"
    if not path.exists():
        log.warning(f"Summary not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


# ── Comparison Diagrams ───────────────────────────────────────────────────────

def _bar_chart(ax, values, labels, colors, title, ylabel, fmt=".4f"):
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.55, zorder=3)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    return bars


def make_comparison_chart(summaries):
    """Create a rich multi-panel comparison figure."""
    sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#0F1117",
        "axes.facecolor": "#1A1D2E",
        "axes.labelcolor": "#CCCCCC",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "text.color": "#FFFFFF",
        "grid.color": "#2A2D3E",
        "axes.edgecolor": "#2A2D3E",
    })

    names   = [m["name"]  for m in MODELS]
    colors  = [m["color"] for m in MODELS]
    markers = [m["marker"] for m in MODELS]

    # Extract metrics
    accuracy   = [s.get("best_validation_metric",            0.0) for s in summaries]
    f1_macro   = [s.get("final_test_metrics", {}).get("f1_macro",   0.0) for s in summaries]
    f1_w       = [s.get("final_test_metrics", {}).get("f1_weighted",0.0) for s in summaries]
    auc        = [s.get("final_test_metrics", {}).get("auc",        0.0) for s in summaries]
    precision  = [s.get("final_test_metrics", {}).get("precision_weighted", 0.0) for s in summaries]
    recall     = [s.get("final_test_metrics", {}).get("recall_weighted",    0.0) for s in summaries]
    params_m   = [s.get("total_parameters", 0) / 1e6 for s in summaries]
    train_hrs  = [s.get("training_time", 0) / 3600 for s in summaries]

    fig = plt.figure(figsize=(22, 18), facecolor="#0F1117")
    gs  = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)

    # ── 1. Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    _bar_chart(ax1, accuracy, names, colors, "Validation Accuracy", "Accuracy")

    # ── 2. F1 (Macro)
    ax2 = fig.add_subplot(gs[0, 1])
    _bar_chart(ax2, f1_macro, names, colors, "F1-Score (Macro)", "F1 Macro")

    # ── 3. AUC
    ax3 = fig.add_subplot(gs[0, 2])
    _bar_chart(ax3, auc, names, colors, "AUC-ROC", "AUC")

    # ── 4. Precision vs Recall scatter
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (p, r, name, color, marker) in enumerate(
            zip(precision, recall, names, colors, markers)):
        ax4.scatter(r, p, s=200, color=color, marker=marker, zorder=5,
                    edgecolors="white", linewidths=1.2)
        ax4.annotate(name, (r, p),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=8.5, color=color)
    ax4.set_xlabel("Recall (Weighted)", fontsize=11)
    ax4.set_ylabel("Precision (Weighted)", fontsize=11)
    ax4.set_title("Precision vs Recall", fontsize=13, fontweight="bold")
    ax4.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1)

    # ── 5. Parameter count
    ax5 = fig.add_subplot(gs[1, 1])
    _bar_chart(ax5, params_m, names, colors, "Model Size (Parameters)", "Millions", fmt=".1f")

    # ── 6. Training time
    ax6 = fig.add_subplot(gs[1, 2])
    _bar_chart(ax6, train_hrs, names, colors, "Training Time", "Hours", fmt=".2f")

    # ── 7. Learning curves (validation accuracy per epoch)
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.set_title("Validation Accuracy Over Epochs", fontsize=13, fontweight="bold", pad=10)
    for i, (summary, model_info) in enumerate(zip(summaries, MODELS)):
        history = summary.get("training_history", [])
        if history:
            epochs = [h.get("epoch", e + 1) for e, h in enumerate(history)]
            val_acc = [h.get("val_metrics", {}).get("accuracy",
                       h.get("val_accuracy", 0.0)) for h in history]
            ax7.plot(epochs, val_acc,
                     color=model_info["color"],
                     marker=model_info["marker"],
                     markevery=max(1, len(epochs) // 10),
                     linewidth=2.2, markersize=6,
                     label=model_info["name"])
    ax7.set_xlabel("Epoch", fontsize=11)
    ax7.set_ylabel("Validation Accuracy", fontsize=11)
    ax7.legend(fontsize=10, framealpha=0.3)
    ax7.grid(linestyle="--", alpha=0.4)

    # ── 8. Radar / spider chart
    ax8 = fig.add_subplot(gs[2, 2], polar=True)
    radar_metrics = ["Accuracy", "F1\nMacro", "AUC", "Precision", "Recall"]
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax8.set_facecolor("#1A1D2E")
    ax8.set_title("Performance Radar", fontsize=13, fontweight="bold", pad=20, color="white")
    plt.xticks(angles[:-1], radar_metrics, color="#AAAAAA", size=9)
    ax8.set_rlabel_position(30)
    ax8.set_ylim(0, 1)

    for i, (summary, model_info) in enumerate(zip(summaries, MODELS)):
        vals = [
            accuracy[i],
            f1_macro[i],
            auc[i],
            precision[i],
            recall[i],
        ]
        vals += vals[:1]
        ax8.plot(angles, vals, linewidth=2, linestyle="solid",
                 color=model_info["color"], label=model_info["name"])
        ax8.fill(angles, vals, alpha=0.10, color=model_info["color"])

    ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.2),
               fontsize=8.5, framealpha=0.3)

    # ── Overall title
    fig.suptitle(
        "Hybrid Mamba-CNN vs Baselines — ISIC 2019 Skin Lesion Classification",
        fontsize=16, fontweight="bold", y=0.995, color="white"
    )

    out = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Comparison chart saved -> {out}")
    return out


def make_training_log_table(summaries):
    """Print and save a compact summary table."""
    header = f"{'Model':<22} {'Accuracy':>10} {'F1 Macro':>10} {'AUC':>8} {'Params(M)':>10} {'Time(h)':>8}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for s, m in zip(summaries, MODELS):
        acc   = s.get("best_validation_metric", 0.0)
        f1m   = s.get("final_test_metrics", {}).get("f1_macro", 0.0)
        auc_v = s.get("final_test_metrics", {}).get("auc", 0.0)
        pm    = s.get("total_parameters", 0) / 1e6
        th    = s.get("training_time", 0) / 3600
        rows.append(f"{m['name']:<22} {acc:>10.4f} {f1m:>10.4f} {auc_v:>8.4f} {pm:>10.2f} {th:>8.3f}")

    table = "\n".join(rows)
    log.info("\n\nFINAL COMPARISON TABLE\n" + table + "\n")

    table_path = SUMMARIES_DIR / "comparison_table.txt"
    table_path.write_text(table)
    log.info(f"Table saved -> {table_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("  Hybrid Mamba-CNN - Model Comparison Run")
    log.info(f"  Started: {datetime.now().isoformat()}")
    log.info("=" * 70)

    summaries = []

    for model_info in MODELS:
        result = run_training(model_info)
        summary = load_summary(model_info["key"])
        if result.get("status") == "failed" or not summary:
            log.warning(f"Using placeholder summary for {model_info['name']}")
            summary = {
                "best_validation_metric": 0.0,
                "total_parameters": 0,
                "training_time": result.get("training_time", 0),
                "final_test_metrics": {},
                "training_history": [],
            }
        else:
            summary.setdefault("training_time", result["training_time"])
        summaries.append(summary)

    log.info("All training runs complete. Generating comparison diagrams...")

    chart_path = make_comparison_chart(summaries)
    make_training_log_table(summaries)

    log.info("=" * 70)
    log.info(f"Training log  : {LOG_FILE}")
    log.info(f"Comparison    : {chart_path}")
    log.info(f"Summary table : {SUMMARIES_DIR / 'comparison_table.txt'}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
