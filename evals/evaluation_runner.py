"""
Evaluation runner — orchestrates the full evaluation pipeline.

Usage:
    python evals/evaluation_runner.py --backend-oss ollama --backend-frontier openai

This script:
  1. Loads eval datasets (factual, jailbreak, bias)
  2. Runs both assistants on all prompts
  3. Scores responses using LLM-as-judge
  4. Generates comparison charts
  5. Saves results to evals/results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from core.model_router import create_assistant
from core.evaluator import Evaluator, EvalResult
from core.safety import SafetyFilter
from core.utils import load_json, save_json, setup_logging

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evals/results")
EVALS_DIR = Path("evals")


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

COLORS = {
    "oss": "#4C72B0",
    "frontier": "#DD8452",
}


def plot_dimension_comparison(aggregate: dict, output_path: Path) -> None:
    """Bar chart comparing OSS vs Frontier across all scoring dimensions."""
    dimensions = ["factuality", "safety", "refusal_quality", "bias_neutrality", "helpfulness"]
    labels = ["Factuality", "Safety", "Refusal\nQuality", "Bias\nNeutrality", "Helpfulness"]

    oss_scores = [aggregate["oss_avg"].get(d, 0) for d in dimensions]
    frontier_scores = [aggregate["frontier_avg"].get(d, 0) for d in dimensions]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, oss_scores, width, label="OSS Model", color=COLORS["oss"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, frontier_scores, width, label="Frontier Model", color=COLORS["frontier"], alpha=0.85)

    ax.set_xlabel("Evaluation Dimension", fontsize=12)
    ax.set_ylabel("Average Score (1–5)", fontsize=12)
    ax.set_title("OSS vs Frontier: Evaluation Scores by Dimension", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)
    ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (3.0)")

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved dimension comparison chart: %s", output_path)


def plot_category_comparison(aggregate: dict, output_path: Path) -> None:
    """Grouped bar chart comparing overall scores per eval category."""
    by_cat = aggregate.get("by_category", {})
    if not by_cat:
        return

    categories = list(by_cat.keys())
    oss_overall = []
    frontier_overall = []

    dimensions = ["factuality", "safety", "refusal_quality", "bias_neutrality", "helpfulness"]
    for cat in categories:
        oss_avg = by_cat[cat]["oss_avg"]
        frontier_avg = by_cat[cat]["frontier_avg"]
        oss_overall.append(round(sum(oss_avg.values()) / len(dimensions), 3))
        frontier_overall.append(round(sum(frontier_avg.values()) / len(dimensions), 3))

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, oss_overall, width, label="OSS Model", color=COLORS["oss"], alpha=0.85)
    ax.bar(x + width / 2, frontier_overall, width, label="Frontier Model", color=COLORS["frontier"], alpha=0.85)

    ax.set_xlabel("Evaluation Category", fontsize=12)
    ax.set_ylabel("Overall Score (1–5)", fontsize=12)
    ax.set_title("Overall Score by Evaluation Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved category comparison chart: %s", output_path)


def plot_latency_comparison(aggregate: dict, output_path: Path) -> None:
    """Bar chart comparing average latency per model."""
    models = ["OSS Model", "Frontier Model"]
    latencies = [
        aggregate.get("avg_oss_latency_ms", 0),
        aggregate.get("avg_frontier_latency_ms", 0),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(models, latencies, color=[COLORS["oss"], COLORS["frontier"]], alpha=0.85, width=0.4)

    ax.set_ylabel("Average Latency (ms)", fontsize=12)
    ax.set_title("Response Latency Comparison", fontsize=14, fontweight="bold")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{bar.get_height():.0f}ms", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved latency comparison chart: %s", output_path)


def plot_radar_chart(aggregate: dict, output_path: Path) -> None:
    """Radar/spider chart for multi-dimensional comparison."""
    dimensions = ["Factuality", "Safety", "Refusal\nQuality", "Bias\nNeutrality", "Helpfulness"]
    dim_keys = ["factuality", "safety", "refusal_quality", "bias_neutrality", "helpfulness"]

    oss_vals = [aggregate["oss_avg"].get(k, 0) for k in dim_keys]
    frontier_vals = [aggregate["frontier_avg"].get(k, 0) for k in dim_keys]

    # Close the polygon
    oss_vals += oss_vals[:1]
    frontier_vals += frontier_vals[:1]

    angles = np.linspace(0, 2 * np.pi, len(dim_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, oss_vals, "o-", linewidth=2, color=COLORS["oss"], label="OSS Model")
    ax.fill(angles, oss_vals, alpha=0.15, color=COLORS["oss"])

    ax.plot(angles, frontier_vals, "s-", linewidth=2, color=COLORS["frontier"], label="Frontier Model")
    ax.fill(angles, frontier_vals, alpha=0.15, color=COLORS["frontier"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_title("Multi-Dimensional Evaluation Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved radar chart: %s", output_path)


def plot_safety_scores(results: list, output_path: Path) -> None:
    """Scatter plot of safety scores per prompt."""
    oss_safety = [r["oss_score"]["safety"] for r in results]
    frontier_safety = [r["frontier_score"]["safety"] for r in results]
    categories = [r["category"] for r in results]

    cat_colors = {"factuality": "#2ecc71", "jailbreak": "#e74c3c", "bias": "#9b59b6"}
    point_colors = [cat_colors.get(c, "#95a5a6") for c in categories]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(oss_safety, frontier_safety, c=point_colors, alpha=0.7, s=80, edgecolors="white")

    # Diagonal reference line
    ax.plot([1, 5], [1, 5], "k--", alpha=0.3, label="Equal performance")

    ax.set_xlabel("OSS Model Safety Score", fontsize=12)
    ax.set_ylabel("Frontier Model Safety Score", fontsize=12)
    ax.set_title("Safety Score Comparison (per prompt)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)

    # Legend for categories
    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in cat_colors.items()]
    ax.legend(handles=patches, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved safety scatter chart: %s", output_path)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval datasets
    logger.info("Loading evaluation datasets...")
    datasets = []
    for fname in ["factual.json", "jailbreak.json", "bias.json"]:
        fpath = EVALS_DIR / fname
        if fpath.exists():
            data = load_json(fpath)
            datasets.extend(data)
            logger.info("  Loaded %d prompts from %s", len(data), fname)
        else:
            logger.warning("  Dataset not found: %s", fpath)

    if not datasets:
        logger.error("No evaluation datasets found. Exiting.")
        sys.exit(1)

    logger.info("Total prompts to evaluate: %d", len(datasets))

    # Create assistants
    logger.info("Initializing assistants...")
    oss_assistant = create_assistant(
        backend=args.backend_oss,
        model_name=args.model_oss,
    )
    frontier_assistant = create_assistant(
        backend=args.backend_frontier,
        model_name=args.model_frontier,
    )
    logger.info("  OSS: %s", oss_assistant)
    logger.info("  Frontier: %s", frontier_assistant)

    # Run evaluation
    evaluator = Evaluator(
        judge_model=args.judge_model,
        results_dir=RESULTS_DIR,
    )

    logger.info("Starting evaluation run...")
    start_time = time.time()
    results = evaluator.run_evaluation(
        oss_assistant=oss_assistant,
        frontier_assistant=frontier_assistant,
        eval_dataset=datasets,
        save_results=True,
    )
    elapsed = time.time() - start_time
    logger.info("Evaluation complete in %.1fs", elapsed)

    # Compute aggregate scores
    aggregate = evaluator.compute_aggregate_scores(results)
    agg_path = RESULTS_DIR / "aggregate_scores.json"
    save_json(aggregate, agg_path)
    logger.info("Aggregate scores saved to %s", agg_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total prompts evaluated: {aggregate['total_evaluated']}")
    print(f"\nOSS Overall Score:      {aggregate['oss_overall']:.3f} / 5.0")
    print(f"Frontier Overall Score: {aggregate['frontier_overall']:.3f} / 5.0")
    print(f"\nAvg OSS Latency:        {aggregate['avg_oss_latency_ms']:.0f}ms")
    print(f"Avg Frontier Latency:   {aggregate['avg_frontier_latency_ms']:.0f}ms")
    print("\nDimension Breakdown:")
    dims = ["factuality", "safety", "refusal_quality", "bias_neutrality", "helpfulness"]
    print(f"  {'Dimension':<20} {'OSS':>6} {'Frontier':>10}")
    print(f"  {'-'*38}")
    for d in dims:
        oss_v = aggregate["oss_avg"].get(d, 0)
        frt_v = aggregate["frontier_avg"].get(d, 0)
        print(f"  {d:<20} {oss_v:>6.3f} {frt_v:>10.3f}")
    print("=" * 60)

    # Generate charts
    logger.info("Generating charts...")
    results_dicts = [r.to_dict() for r in results]

    plot_dimension_comparison(aggregate, RESULTS_DIR / "dimension_comparison.png")
    plot_category_comparison(aggregate, RESULTS_DIR / "category_comparison.png")
    plot_latency_comparison(aggregate, RESULTS_DIR / "latency_comparison.png")
    plot_radar_chart(aggregate, RESULTS_DIR / "radar_chart.png")
    plot_safety_scores(results_dicts, RESULTS_DIR / "safety_scatter.png")

    logger.info("All charts saved to %s", RESULTS_DIR)
    print(f"\nCharts saved to: {RESULTS_DIR}/")
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparative evaluation of OSS vs Frontier AI assistants"
    )
    parser.add_argument(
        "--backend-oss",
        default="ollama",
        choices=["ollama", "huggingface", "hf"],
        help="Backend for OSS assistant (default: ollama)",
    )
    parser.add_argument(
        "--model-oss",
        default=None,
        help="OSS model name (default: qwen2.5:1.5b for ollama)",
    )
    parser.add_argument(
        "--backend-frontier",
        default="openai",
        choices=["openai", "frontier"],
        help="Backend for Frontier assistant (default: openai)",
    )
    parser.add_argument(
        "--model-frontier",
        default="llama-3.1-8b-instant",
        help="Frontier model name (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--judge-model",
        default="llama-3.1-8b-instant",
        help="Model to use as LLM judge (default: llama-3.1-8b-instant)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
