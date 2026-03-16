#!/usr/bin/env python
"""Plot utility gap vs Beta(t,t) distribution parameter at a fixed alpha level.

Sweeps the symmetric Beta(t,t) parameter t log-uniformly from near-bimodal (t→0)
to near-point-mass at 1/2 (t→∞). For each t, computes:

    utility gap = U_optimal(alpha) - U_optimal(0)

where U is the fraction of true positives identified (%).

Usage:
    python scripts/plot_utility_gap_vs_dist_param.py
    python scripts/plot_utility_gap_vs_dist_param.py --alpha 0.25 --beta 0.35 --n-sims 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, "src")
from optimal_screening.analysis import compute_optimal_screening_curve


def main():
    parser = argparse.ArgumentParser(
        description="Utility gap vs Beta(t,t) parameter at fixed alpha"
    )
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Fixed screening budget α (default: 0.25)")
    parser.add_argument("--beta", type=float, default=0.35,
                        help="Treatment budget β (default: 0.35)")
    parser.add_argument("--n-sims", type=int, default=3,
                        help="Simulations per parameter value (default: 3)")
    parser.add_argument("--n-params", type=int, default=50,
                        help="Number of t values on the log grid (default: 50)")
    parser.add_argument("--t-min", type=float, default=0.02,
                        help="Minimum Beta parameter (default: 0.02, near-bimodal)")
    parser.add_argument("--t-max", type=float, default=200.0,
                        help="Maximum Beta parameter (default: 200.0, near-point-mass)")
    parser.add_argument("--output", type=str, default="runs/utility_gap_vs_dist_param.png",
                        help="Output plot path")
    args = parser.parse_args()

    if args.alpha > args.beta:
        raise ValueError(f"alpha={args.alpha} must be <= beta={args.beta}")

    alpha = args.alpha
    beta = args.beta
    n_sims = args.n_sims

    t_values = np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.n_params)

    print(f"Utility gap vs Beta(t,t) parameter")
    print(f"  α={alpha:.0%}, β={beta:.0%}, {n_sims} sims per point, {len(t_values)} t values")
    print(f"  t range: [{args.t_min}, {args.t_max}] (log scale)")
    print()

    gaps_mean = []
    gaps_std = []

    for t_idx, t in enumerate(t_values):
        gap_runs = []
        for sim_idx in range(n_sims):
            result = compute_optimal_screening_curve(
                rows=[],
                outcome_col="_outcome",
                strata_features=[],
                beta=beta,
                alpha_quantiles=[0.0, alpha],
                seed=sim_idx,
                simulation=(t, t),
            )
            total_pos = result["total_positive"]
            if total_pos > 0:
                u_no_screen = result["true_positives"][0] / total_pos * 100
                u_screen = result["true_positives"][1] / total_pos * 100
                gap_runs.append(u_screen - u_no_screen)
            else:
                gap_runs.append(0.0)

        gaps_mean.append(np.mean(gap_runs))
        gaps_std.append(np.std(gap_runs))

        if (t_idx + 1) % 10 == 0 or t_idx == len(t_values) - 1:
            print(f"  [{t_idx + 1:3d}/{len(t_values)}]  t={t:.4f}  gap={gaps_mean[-1]:.2f} ± {gaps_std[-1]:.2f} pp")

    gaps_mean = np.array(gaps_mean)
    gaps_std = np.array(gaps_std)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_values, gaps_mean, linewidth=2.5, color="steelblue", marker="o", markersize=4, zorder=2)
    ax.fill_between(t_values, gaps_mean - gaps_std, gaps_mean + gaps_std,
                    alpha=0.2, color="steelblue", zorder=1)

    # Reference lines for the named presets
    preset_refs = [
        (0.1,  "bimodal (0.1, 0.1)",  "orange"),
        (1.0,  "uniform (1, 1)",       "blue"),
        (10.0, "gaussian (10, 10)",    "green"),
    ]
    for t_ref, label, color in preset_refs:
        ax.axvline(x=t_ref, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Beta distribution parameter  t  [Beta(t, t)]", fontsize=13)
    ax.set_ylabel("Utility Gap (percentage points)", fontsize=13)
    ax.set_title(
        f"Utility Gap: Optimal vs No Screening\n"
        f"α={alpha:.0%}, β={beta:.0%}, {n_sims} sim{'s' if n_sims > 1 else ''} per point",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(args.t_min, args.t_max)
    ax.set_ylim(bottom=0)

    # Annotate extremes
    ax.text(args.t_min * 1.3, ax.get_ylim()[1] * 0.95, "← bimodal\n(0 / 1 risk)",
            fontsize=9, va="top", color="gray")
    ax.text(args.t_max * 0.5, ax.get_ylim()[1] * 0.95, "point mass\nat 0.5 →",
            fontsize=9, va="top", ha="right", color="gray")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
