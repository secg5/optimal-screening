#!/usr/bin/env python
"""Run multiple simulations with random uniform probabilities and plot mean ± std."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, 'src')
from optimal_screening.analysis import compute_optimal_screening_curve

# Parameters
CSV_PATH = "data/acs_income_sample.csv"
OUTCOME_COL = "PINCP > 50k"
STRATA_FEATURES = ["AGEP", "SEX"]
BETA = 0.3
N_SIMULATIONS = 100

print(f"Running {N_SIMULATIONS} simulations with β={BETA:.1%}...")

# Load data
df = pd.read_csv(CSV_PATH)
rows = df.to_dict("records")

# Run simulations
all_results = []
for sim_idx in range(N_SIMULATIONS):
    # Run optimal screening with seed for this simulation
    results = compute_optimal_screening_curve(
        rows=rows,
        outcome_col=OUTCOME_COL,
        strata_features=STRATA_FEATURES,
        beta=BETA,
        seed=sim_idx,
    )

    all_results.append({
        'alpha_values': results['alpha_values'],
        'true_positives': results['true_positives']
    })

    if (sim_idx + 1) % 10 == 0:
        print(f"  Completed {sim_idx + 1}/{N_SIMULATIONS} simulations")

# Aggregate results
alpha_values = all_results[0]['alpha_values']
tp_matrix = np.array([r['true_positives'] for r in all_results])

# Compute statistics
tp_mean = tp_matrix.mean(axis=0)
tp_std = tp_matrix.std(axis=0)
tp_min = tp_matrix.min(axis=0)
tp_max = tp_matrix.max(axis=0)

# Print summary
print(f"\nResults (mean ± std):")
for i, alpha in enumerate(alpha_values):
    print(f"  α={alpha:5.1%}: {tp_mean[i]:5.1f} ± {tp_std[i]:4.1f}  (range: {tp_min[i]}-{tp_max[i]})")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

alpha_pct = [a * 100 for a in alpha_values]

# Plot mean with shaded std
ax.plot(alpha_pct, tp_mean, marker='o', linewidth=2, markersize=6,
        label=f'Mean (n={N_SIMULATIONS})', color='blue')
ax.fill_between(alpha_pct, tp_mean - tp_std, tp_mean + tp_std,
                alpha=0.3, color='blue', label='± 1 std')

# Reference lines
total_positive = len([r for r in rows if r[OUTCOME_COL]])
ax.axhline(y=tp_mean[0], color='red', linestyle='--', alpha=0.7,
           label=f'No screening (α=0, TP={tp_mean[0]:.0f})')
ax.axhline(y=total_positive, color='green', linestyle='--', alpha=0.7,
           label=f'Perfect (TP={total_positive})')

ax.set_xlabel('Screening Budget α (%)', fontsize=12)
ax.set_ylabel('True Positives Correctly Identified', fontsize=12)
ax.set_title(f'Optimal Screening Curve ({N_SIMULATIONS} Simulations)\nTreatment Budget β = {BETA:.1%}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, BETA * 100)

# Save
output_path = Path("runs/beta_0.6_simulations.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {output_path}")
