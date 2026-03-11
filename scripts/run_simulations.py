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
from scipy.stats import beta as beta_dist


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
    "delta_half": {"color": "red", "color_pastel": "#ffadad", "marker": "D"},
}


def main():
    parser = argparse.ArgumentParser(description="Run simulated screening curves for Beta distribution presets")
    parser.add_argument("--beta", type=float, default=0.35, help="Treatment budget (default: 0.35)")
    parser.add_argument("--n-sims", type=int, default=10, help="Number of simulations per preset (default: 10)")
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
        params = RISK_PRESETS[preset]
        if isinstance(params, tuple):
            print(f"  [{preset}] Beta({params[0]}, {params[1]})")
        else:
            print(f"  [{preset}] Point mass at {params}")

        optimal_tp_runs = []
        intuitive_tp_runs = []
        random_tp_runs = []
        optimal_prec_runs = []
        intuitive_prec_runs = []
        random_prec_runs = []
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

            random_baseline = compute_random_screening_curve(
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
            total_pos_rnd = random_baseline["total_positive"]
            n_samp_opt = optimal["total_samples"]
            n_samp_int = intuitive["total_samples"]
            n_samp_rnd = random_baseline["total_samples"]
            treated_opt = beta * n_samp_opt
            treated_int = beta * n_samp_int
            treated_rnd = beta * n_samp_rnd

            if total_pos_opt > 0:
                optimal_tp_runs.append([tp / total_pos_opt * 100 for tp in optimal["true_positives"]])
            else:
                optimal_tp_runs.append([0.0] * len(optimal["true_positives"]))

            if total_pos_int > 0:
                intuitive_tp_runs.append([tp / total_pos_int * 100 for tp in intuitive["true_positives"]])
            else:
                intuitive_tp_runs.append([0.0] * len(intuitive["true_positives"]))

            if total_pos_rnd > 0:
                random_tp_runs.append([tp / total_pos_rnd * 100 for tp in random_baseline["true_positives"]])
            else:
                random_tp_runs.append([0.0] * len(random_baseline["true_positives"]))

            optimal_prec_runs.append(
                [tp / treated_opt * 100 if treated_opt > 0 else 0.0 for tp in optimal["true_positives"]]
            )
            intuitive_prec_runs.append(
                [tp / treated_int * 100 if treated_int > 0 else 0.0 for tp in intuitive["true_positives"]]
            )
            random_prec_runs.append(
                [tp / treated_rnd * 100 if treated_rnd > 0 else 0.0 for tp in random_baseline["true_positives"]]
            )

            if (sim_idx + 1) % 25 == 0:
                print(f"    Completed {sim_idx + 1}/{n_sims}")

        all_preset_results[preset] = {
            "alpha_values": alpha_values,
            "optimal_tp": np.array(optimal_tp_runs),
            "intuitive_tp": np.array(intuitive_tp_runs),
            "random_tp": np.array(random_tp_runs),
            "optimal_prec": np.array(optimal_prec_runs),
            "intuitive_prec": np.array(intuitive_prec_runs),
            "random_prec": np.array(random_prec_runs),
        }

    # --- Plots ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_pct = [a * 100 for a in all_preset_results[next(iter(all_preset_results))]["alpha_values"]]

    def _format_preset(preset):
        params = RISK_PRESETS[preset]
        return f"({params[0]}, {params[1]})" if isinstance(params, tuple) else f"δ({params})"

    def _style_ax(ax, title, y_values=None, ylabel="True Positives Identified (%)"):
        ax.set_xlabel("Screening Budget α (%)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, beta * 100)
        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            margin = (y_max - y_min) * 0.15
            ax.set_ylim(max(0, y_min - margin), min(105, y_max + margin))
        else:
            ax.set_ylim(0, 105)

    # --- Plot 1: Optimal vs Intuitive ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    y_vals_1 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        int_mean = data["intuitive_tp"].mean(axis=0)
        y_vals_1.extend([int_mean.min(), int_mean.max()])
        ax1.plot(alpha_pct, int_mean, marker=style["marker"], linewidth=2, markersize=4,
                 color=style["color_pastel"], label=f"Intuitive – {preset}", zorder=1)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_tp"].mean(axis=0)
        label = _format_preset(preset)
        y_vals_1.extend([opt_mean.min(), opt_mean.max()])
        ax1.plot(alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"Optimal – {preset} {label}", zorder=2)

    _style_ax(ax1, f"Optimal vs Intuitive Screening ({n_sims} sims, β={beta:.0%})", y_vals_1)
    fig1.tight_layout()
    path1 = output_path.with_name("optimal_vs_intuitive.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {path1}")

    # --- Plot 2: Optimal vs Random ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    y_vals_2 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        rnd_mean = data["random_tp"].mean(axis=0)
        y_vals_2.extend([rnd_mean.min(), rnd_mean.max()])
        ax2.plot(alpha_pct, rnd_mean, marker=style["marker"], linewidth=2, markersize=4,
                 linestyle="--", color=style["color_pastel"], label=f"Random – {preset}", zorder=1)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_tp"].mean(axis=0)
        label = _format_preset(preset)
        y_vals_2.extend([opt_mean.min(), opt_mean.max()])
        ax2.plot(alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"Optimal – {preset} {label}", zorder=2)

    _style_ax(ax2, f"Optimal vs Random Screening ({n_sims} sims, β={beta:.0%})", y_vals_2)
    fig2.tight_layout()
    path2 = output_path.with_name("optimal_vs_random.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path2}")

    # --- Plot 3: Optimal vs No Screening ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    y_vals_3 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        no_screen_val = data["optimal_tp"].mean(axis=0)[0]
        y_vals_3.append(no_screen_val)
        ax3.axhline(y=no_screen_val, color=style["color"], linestyle=":", linewidth=1.5, alpha=0.6,
                     label=f"No screening – {preset} ({no_screen_val:.1f}%)", zorder=1)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_tp"].mean(axis=0)
        label = _format_preset(preset)
        y_vals_3.extend([opt_mean.min(), opt_mean.max()])
        ax3.plot(alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"Optimal – {preset} {label}", zorder=2)

    _style_ax(ax3, f"Optimal vs No Screening ({n_sims} sims, β={beta:.0%})", y_vals_3)
    fig3.tight_layout()
    path3 = output_path.with_name("optimal_vs_no_screening.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path3}")

    prec_ylabel = "Precision = TP / Treated (%)"

    # --- Plot 4: Precision – Optimal vs Intuitive ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    y_vals_4 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        int_mean = data["intuitive_prec"].mean(axis=0)
        y_vals_4.extend([int_mean.min(), int_mean.max()])
        ax4.plot(alpha_pct, int_mean, marker=style["marker"], linewidth=2, markersize=4,
                 color=style["color_pastel"], label=f"Intuitive – {preset}", zorder=1)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_prec"].mean(axis=0)
        label = _format_preset(preset)
        y_vals_4.extend([opt_mean.min(), opt_mean.max()])
        ax4.plot(alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"Optimal – {preset} {label}", zorder=2)

    _style_ax(ax4, f"Precision: Optimal vs Intuitive ({n_sims} sims, β={beta:.0%})", y_vals_4, ylabel=prec_ylabel)
    fig4.tight_layout()
    path4 = output_path.with_name("precision_optimal_vs_intuitive.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path4}")

    # --- Plot 5: Precision – Optimal vs Random ---
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    y_vals_5 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        rnd_mean = data["random_prec"].mean(axis=0)
        y_vals_5.extend([rnd_mean.min(), rnd_mean.max()])
        ax5.plot(alpha_pct, rnd_mean, marker=style["marker"], linewidth=2, markersize=4,
                 linestyle="--", color=style["color_pastel"], label=f"Random – {preset}", zorder=1)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_prec"].mean(axis=0)
        label = _format_preset(preset)
        y_vals_5.extend([opt_mean.min(), opt_mean.max()])
        ax5.plot(alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"Optimal – {preset} {label}", zorder=2)

    _style_ax(ax5, f"Precision: Optimal vs Random ({n_sims} sims, β={beta:.0%})", y_vals_5, ylabel=prec_ylabel)
    fig5.tight_layout()
    path5 = output_path.with_name("precision_optimal_vs_random.png")
    fig5.savefig(path5, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path5}")

    # --- Plot 6: Precision – Optimal vs No Screening ---
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    y_vals_6 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        no_screen_val = data["optimal_prec"].mean(axis=0)[0]
        y_vals_6.append(no_screen_val)
        ax6.axhline(y=no_screen_val, color=style["color"], linestyle=":", linewidth=1.5, alpha=0.6,
                     label=f"No screening – {preset} ({no_screen_val:.1f}%)", zorder=1)
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_prec"].mean(axis=0)
        label = _format_preset(preset)
        y_vals_6.extend([opt_mean.min(), opt_mean.max()])
        ax6.plot(alpha_pct, opt_mean, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"Optimal – {preset} {label}", zorder=2)

    _style_ax(ax6, f"Precision: Optimal vs No Screening ({n_sims} sims, β={beta:.0%})", y_vals_6, ylabel=prec_ylabel)
    fig6.tight_layout()
    path6 = output_path.with_name("precision_optimal_vs_no_screening.png")
    fig6.savefig(path6, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path6}")

    # --- Plot 7: Utility Gap (Optimal vs No Screening) ---
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    y_vals_7 = []
    for preset, data in all_preset_results.items():
        style = PRESET_STYLES[preset]
        opt_mean = data["optimal_tp"].mean(axis=0)
        no_screen_val = opt_mean[0]
        gap = opt_mean - no_screen_val
        label = _format_preset(preset)
        y_vals_7.extend([gap.min(), gap.max()])
        ax7.plot(alpha_pct, gap, marker=style["marker"], linewidth=2.5, markersize=5,
                 color=style["color"], label=f"{preset} {label}")

    ax7.set_xlabel("Screening Budget α (%)", fontsize=13)
    ax7.set_ylabel("Utility Gap (percentage points)", fontsize=13)
    ax7.set_title(f"Utility Gap: Optimal vs No Screening ({n_sims} sims, β={beta:.0%})", fontsize=14, fontweight="bold")
    ax7.legend(fontsize=10, loc="upper left")
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, beta * 100)
    ax7.set_ylim(bottom=0)
    fig7.tight_layout()
    path7 = output_path.with_name("utility_gap_optimal_vs_no_screening.png")
    fig7.savefig(path7, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path7}")

    # --- Risk distribution densities plot ---
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 500)

    for preset, params in RISK_PRESETS.items():
        style = PRESET_STYLES[preset]
        if isinstance(params, tuple):
            a_param, b_param = params
            y = beta_dist.pdf(x, a_param, b_param)
            ax_dist.plot(x, y, linewidth=2.5, color=style["color"], label=f"{preset} — Beta({a_param}, {b_param})")
        else:
            ax_dist.axvline(x=params, color=style["color"], linewidth=2.5, label=f"{preset} — δ({params})")

    ax_dist.set_xlabel("Risk score", fontsize=13)
    ax_dist.set_ylabel("Density", fontsize=13)
    ax_dist.set_title("Risk Score Distributions", fontsize=15, fontweight="bold")
    ax_dist.legend(fontsize=11)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_ylim(0, 5)

    dist_path = output_path.with_name("risk_distributions.png")
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    print(f"Distribution plot saved to {dist_path}")


if __name__ == "__main__":
    main()
