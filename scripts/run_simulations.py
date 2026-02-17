#!/usr/bin/env python
"""Run simulated screening curves for each risk distribution preset and plot together.

Usage:
    python scripts/run_simulations.py
    python scripts/run_simulations.py --beta 0.5 --n-sims 5 --output runs/simulations.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, "src")
from optimal_screening.analysis import (
    RISK_PRESETS,
    compute_intuitive_optimal_curve,
    compute_optimal_screening_curve,
    compute_random_screening_curve,
)


PRESET_STYLES = {
    "uniform": {"color": "blue", "color_pastel": "#a0c4ff", "marker": "o"},
    "bimodal": {"color": "orange", "color_pastel": "#ffd6a5", "marker": "s"},
    "gaussian": {"color": "green", "color_pastel": "#caffbf", "marker": "^"},
}


def main():
    parser = argparse.ArgumentParser(description="Run simulated screening curves for Beta distribution presets")
    parser.add_argument("--beta", type=float, default=0.3, help="Treatment budget (default: 0.3)")
    parser.add_argument("--n-sims", type=int, default=100, help="Number of simulations per preset (default: 100)")
    parser.add_argument("--output", type=str, default="runs/simulations.png", help="Output plot path")
    args = parser.parse_args()

    beta = args.beta
    n_sims = args.n_sims

    print(f"Running {n_sims} simulations per preset with β={beta:.1%}...")
    print(f"Presets: {list(RISK_PRESETS.keys())}")
    print()

    # --- Optimal + intuitive optimal per preset ---
    all_preset_results = {}

    for preset in RISK_PRESETS:
        print(f"  [{preset}] Beta({RISK_PRESETS[preset][0]}, {RISK_PRESETS[preset][1]})")

        optimal_tp_runs = []
        intuitive_tp_runs = []
        alpha_values = None

        for sim_idx in range(n_sims):
            optimal = compute_optimal_screening_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                seed=sim_idx,
                simulation=preset,
            )

            intuitive = compute_intuitive_optimal_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                seed=sim_idx,
                simulation=preset,
            )

            if alpha_values is None:
                alpha_values = optimal["alpha_values"]

            total_pos_opt = optimal["total_positive"]
            total_pos_int = intuitive["total_positive"]

            if total_pos_opt > 0:
                optimal_tp_runs.append([tp / total_pos_opt * 100 for tp in optimal["true_positives"]])
            else:
                optimal_tp_runs.append([0.0] * len(optimal["true_positives"]))

            if total_pos_int > 0:
                intuitive_tp_runs.append([tp / total_pos_int * 100 for tp in intuitive["true_positives"]])
            else:
                intuitive_tp_runs.append([0.0] * len(intuitive["true_positives"]))

            if (sim_idx + 1) % 25 == 0:
                print(f"    Completed {sim_idx + 1}/{n_sims}")

        all_preset_results[preset] = {
            "alpha_values": alpha_values,
            "optimal_tp": np.array(optimal_tp_runs),
            "intuitive_tp": np.array(intuitive_tp_runs),
        }

    # --- Single random screening baseline (uses uniform population) ---
    print("  [random baseline]")
    random_tp_runs = []
    for sim_idx in range(n_sims):
        random_baseline = compute_random_screening_curve(
            rows=[],
            outcome_col="_outcome",
            strata_features=[],
            beta=beta,
            seed=sim_idx,
            simulation="uniform",
        )

        total_pos = random_baseline["total_positive"]
        if total_pos > 0:
            random_tp_runs.append([tp / total_pos * 100 for tp in random_baseline["true_positives"]])
        else:
            random_tp_runs.append([0.0] * len(random_baseline["true_positives"]))

        if (sim_idx + 1) % 25 == 0:
            print(f"    Completed {sim_idx + 1}/{n_sims}")

    random_tp_matrix = np.array(random_tp_runs)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

    alpha_pct = [a * 100 for a in all_preset_results[next(iter(all_preset_results))]["alpha_values"]]

    # Plot intuitive optimal first (background)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        int_mean = data["intuitive_tp"].mean(axis=0)

        ax.plot(
            alpha_pct,
            int_mean,
            marker=style["marker"],
            linewidth=2,
            markersize=4,
            color=style["color_pastel"],
            label=f"Intuitive – {preset}",
            zorder=1,
        )

    # Plot optimal screening (foreground)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        a_param, b_param = RISK_PRESETS[preset]

        opt_mean = data["optimal_tp"].mean(axis=0)
        opt_std = data["optimal_tp"].std(axis=0)

        ax.plot(
            alpha_pct,
            opt_mean,
            marker=style["marker"],
            linewidth=2.5,
            markersize=6,
            color=style["color"],
            label=f"Optimal – {preset} ({a_param}, {b_param})",
            zorder=3,
        )
        ax.fill_between(alpha_pct, opt_mean - opt_std, opt_mean + opt_std, alpha=0.15, color=style["color"], zorder=2)

    # Single random baseline
    rand_mean = random_tp_matrix.mean(axis=0)
    ax.plot(
        alpha_pct,
        rand_mean,
        linestyle="--",
        linewidth=2,
        color="gray",
        alpha=0.8,
        label="Random screening",
        zorder=2,
    )

    # No-screening reference
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        no_screen_val = data["optimal_tp"].mean(axis=0)[0]
        ax.axhline(
            y=no_screen_val,
            color=style["color"],
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label=f"No screening – {preset} ({no_screen_val:.1f}%)",
        )

    # Perfect screening
    ax.axhline(y=100, color="black", linestyle="--", linewidth=1.5, alpha=0.6, label="Perfect (100%)")

    ax.set_xlabel("Screening Budget α (%)", fontsize=13)
    ax.set_ylabel("True Positives Identified (%)", fontsize=13)
    ax.set_title(
        f"Optimal Screening Across Risk Distributions ({n_sims} simulations)\n"
        f"Treatment Budget β = {beta:.0%}",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, beta * 100)
    ax.set_ylim(0, 105)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
