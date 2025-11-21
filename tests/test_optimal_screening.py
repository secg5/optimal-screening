"""Unit tests for optimal screening curve computation."""

from __future__ import annotations

import numpy as np

from optimal_screening.analysis import compute_optimal_screening_curve


class TestOptimalScreeningCurveUniform:
    """Test compute_optimal_screening_curve with uniform distribution.

    When risks are drawn from U(0,1), we can analytically compute expected band properties.
    """

    def analytical_avg_risk_band3(self, beta: float, alpha: float) -> float:
        """Compute analytical average risk for Band 3 under uniform distribution.

        For uniform U(0,1) risks sorted descending, Band 3's average risk is:
        avg_risk_band3 = (1 - β - α/2) / (1 - α)

        Derivation:
        - Band 1: positions [0, β-α), risks [1, 1-β+α)
        - Band 2: positions [β-α, β-α·r̄₃), where r̄₃ = avg_risk_band3
        - Band 3: positions [β-α·r̄₃, β+α(1-r̄₃)), risks uniform in this range
        - For uniform, risk at position x is (1-x)
        - Band 3 risks: from (1-(β-α·r̄₃)) to (1-(β+α(1-r̄₃)))
        - Average: [(1-β+α·r̄₃) + (1-β-α(1-r̄₃))] / 2 = (2-2β-α+2α·r̄₃) / 2
        - Solving: r̄₃ = (1 - β - α/2) / (1 - α)
        """
        if alpha >= 1.0:
            return 0.0
        return (1 - beta - alpha / 2) / (1 - alpha)

    def analytical_band_sizes(self, beta: float, alpha: float, avg_risk_band3: float) -> dict[str, float]:
        """Compute analytical band population proportions."""
        band1_size = beta - alpha
        band2_size = alpha * (1 - avg_risk_band3)
        band3_size = alpha
        band4_size = 1 - beta - band2_size

        return {
            "band1_pop_prop": band1_size,
            "band2_pop_prop": band2_size,
            "band3_pop_prop": band3_size,
            "band4_pop_prop": band4_size,
        }

    def analytical_band_integrals(self, beta: float, alpha: float, avg_risk_band3: float) -> dict[str, float]:
        """Compute analytical integrals ∫ risk · f(risk) d(risk) for each band.

        For uniform distribution, f(risk) is constant = 1.
        Position x ∈ [0,1] has risk = 1-x.
        Integral from position a to b: ∫[a,b] (1-x) dx = x - x²/2 |[a,b]
        """
        # Band 1: [0, β-α)
        b1_end = beta - alpha
        band1_integral = (b1_end - b1_end**2 / 2) - (0 - 0)

        # Band 2: [β-α, β-α·r̄₃)
        b2_start = beta - alpha
        b2_end = beta - alpha * avg_risk_band3
        band2_integral = (b2_end - b2_end**2 / 2) - (b2_start - b2_start**2 / 2)

        # Band 3: [β-α·r̄₃, β+α(1-r̄₃))
        b3_start = beta - alpha * avg_risk_band3
        b3_end = beta + alpha * (1 - avg_risk_band3)
        band3_integral = (b3_end - b3_end**2 / 2) - (b3_start - b3_start**2 / 2)

        # Band 4: [β+α(1-r̄₃), 1]
        b4_start = beta + alpha * (1 - avg_risk_band3)
        b4_end = 1.0
        band4_integral = (b4_end - b4_end**2 / 2) - (b4_start - b4_start**2 / 2)

        return {
            "band1_integral": band1_integral,
            "band2_integral": band2_integral,
            "band3_integral": band3_integral,
            "band4_integral": band4_integral,
        }

    def test_uniform_band_structure_single_alpha(self, params):
        """Test band structure for a single alpha value with uniform distribution."""
        np.random.seed(42)

        beta = params["beta"]
        alpha = 0.3

        # Generate data with uniform risks
        n = params["n_samples"]
        risks = np.random.uniform(0, 1, n)
        outcomes = np.random.binomial(1, risks, n)  # Outcomes match risk probabilities

        # Create dataset
        rows = [{"risk": float(risks[i]), "outcome": bool(outcomes[i]), "stratum": 0} for i in range(n)]

        # Compute optimal screening curve
        result = compute_optimal_screening_curve(
            rows=rows,
            outcome_col="outcome",
            strata_features=["stratum"],  # Dummy stratum (not used with custom risk)
            beta=beta,
            alpha_quantiles=[alpha],
            use_custom_risk_col="risk",
        )

        # Get computed values
        band_info = result["band_info"][0]

        # Analytical expectations
        expected_avg_risk = self.analytical_avg_risk_band3(beta, alpha)
        expected_sizes = self.analytical_band_sizes(beta, alpha, expected_avg_risk)
        expected_integrals = self.analytical_band_integrals(beta, alpha, expected_avg_risk)

        # Verify average risk in Band 3
        assert abs(band_info["avg_risk_band3"] - expected_avg_risk) < params["tolerance"], (
            f"Band 3 avg risk: expected {expected_avg_risk:.4f}, got {band_info['avg_risk_band3']:.4f}"
        )

        # Verify band population proportions
        for band_name in ["band1", "band2", "band3", "band4"]:
            key = f"{band_name}_pop_prop"
            expected = expected_sizes[key]
            actual = band_info[key]
            assert abs(actual - expected) < params["tolerance"], f"{key}: expected {expected:.4f}, got {actual:.4f}"

        # Verify band integrals (within tolerance)
        for band_name in ["band1", "band2", "band3", "band4"]:
            key = f"{band_name}_integral"
            expected = expected_integrals[key]
            actual = band_info[key]
            # Integrals have higher variance, use 3x tolerance
            assert abs(actual - expected) < 3 * params["tolerance"], (
                f"{key}: expected {expected:.4f}, got {actual:.4f}"
            )

        print(f"\n✓ Single alpha test passed (α={alpha}, β={beta})")
        print(f"  Expected avg_risk_band3: {expected_avg_risk:.4f}")
        print(f"  Actual avg_risk_band3:   {band_info['avg_risk_band3']:.4f}")
        print(f"  Band sizes (expected): {expected_sizes}")
        print(f"  Band sizes (actual):   {{")
        for k in expected_sizes:
            print(f"    '{k}': {band_info[k]:.4f},")
        print("  }")

    def test_uniform_distribution_multiple_simulations(self, params):
        """Run 100 simulations with uniform distribution and verify band structure."""
        beta = params["beta"]
        alpha_values = params["alpha_values"]
        n_simulations = params["n_simulations"]
        n = params["n_samples"]

        results_summary = []

        for sim_idx in range(n_simulations):
            np.random.seed(sim_idx)

            # Generate data with uniform risks
            risks = np.random.uniform(0, 1, n)
            outcomes = np.random.binomial(1, risks, n)

            rows = [{"risk": float(risks[i]), "outcome": bool(outcomes[i]), "stratum": 0} for i in range(n)]

            # Compute optimal screening curve
            result = compute_optimal_screening_curve(
                rows=rows,
                outcome_col="outcome",
                strata_features=["stratum"],
                beta=beta,
                alpha_quantiles=alpha_values,
                use_custom_risk_col="risk",
            )

            results_summary.append(result)

        # Aggregate results across simulations
        for alpha_idx, alpha in enumerate(alpha_values):
            if alpha > beta:
                continue

            # Collect values across all simulations
            avg_risks = [res["band_info"][alpha_idx]["avg_risk_band3"] for res in results_summary]
            band1_props = [res["band_info"][alpha_idx]["band1_pop_prop"] for res in results_summary]
            band2_props = [res["band_info"][alpha_idx]["band2_pop_prop"] for res in results_summary]
            band3_props = [res["band_info"][alpha_idx]["band3_pop_prop"] for res in results_summary]
            band4_props = [res["band_info"][alpha_idx]["band4_pop_prop"] for res in results_summary]

            # Compute mean across simulations
            mean_avg_risk = np.mean(avg_risks)
            mean_band1_prop = np.mean(band1_props)
            mean_band2_prop = np.mean(band2_props)
            mean_band3_prop = np.mean(band3_props)
            mean_band4_prop = np.mean(band4_props)

            # Analytical expectations
            expected_avg_risk = self.analytical_avg_risk_band3(beta, alpha)
            expected_sizes = self.analytical_band_sizes(beta, alpha, expected_avg_risk)

            # With 100 simulations and large n, errors should be very small
            simulation_tolerance = params["tolerance"] / 3  # Tighter with averaging

            # Verify average risk
            assert abs(mean_avg_risk - expected_avg_risk) < simulation_tolerance, (
                f"Alpha={alpha}: Band 3 avg risk mean over {n_simulations} sims: "
                f"expected {expected_avg_risk:.4f}, got {mean_avg_risk:.4f}"
            )

            # Verify band proportions
            assert abs(mean_band1_prop - expected_sizes["band1_pop_prop"]) < simulation_tolerance
            assert abs(mean_band2_prop - expected_sizes["band2_pop_prop"]) < simulation_tolerance
            assert abs(mean_band3_prop - expected_sizes["band3_pop_prop"]) < simulation_tolerance
            assert abs(mean_band4_prop - expected_sizes["band4_pop_prop"]) < simulation_tolerance

            print(f"\n✓ Alpha={alpha:.1f} passed ({n_simulations} simulations)")
            print(f"  Mean avg_risk_band3: {mean_avg_risk:.4f} (expected: {expected_avg_risk:.4f})")
            print(f"  Mean band sizes:")
            print(f"    Band 1: {mean_band1_prop:.4f} (expected: {expected_sizes['band1_pop_prop']:.4f})")
            print(f"    Band 2: {mean_band2_prop:.4f} (expected: {expected_sizes['band2_pop_prop']:.4f})")
            print(f"    Band 3: {mean_band3_prop:.4f} (expected: {expected_sizes['band3_pop_prop']:.4f})")
            print(f"    Band 4: {mean_band4_prop:.4f} (expected: {expected_sizes['band4_pop_prop']:.4f})")

    def test_band_mass_conservation(self, params):
        """Verify that band population proportions sum to 1."""
        np.random.seed(123)

        n = params["n_samples"]
        risks = np.random.uniform(0, 1, n)
        outcomes = np.random.binomial(1, risks, n)

        rows = [{"risk": float(risks[i]), "outcome": bool(outcomes[i]), "stratum": 0} for i in range(n)]

        result = compute_optimal_screening_curve(
            rows=rows,
            outcome_col="outcome",
            strata_features=["stratum"],
            beta=params["beta"],
            alpha_quantiles=params["alpha_values"],
            use_custom_risk_col="risk",
        )

        for band_info in result["band_info"]:
            total_mass = (
                band_info["band1_pop_prop"]
                + band_info["band2_pop_prop"]
                + band_info["band3_pop_prop"]
                + band_info["band4_pop_prop"]
            )
            assert abs(total_mass - 1.0) < 1e-6, (
                f"Band proportions don't sum to 1: {total_mass:.6f} (alpha={band_info['alpha']})"
            )

        print("\n✓ Band mass conservation verified for all alpha values")

    def test_alpha_zero_edge_case(self, params):
        """Test edge case where α=0 (no screening)."""
        np.random.seed(456)

        n = params["n_samples"]
        risks = np.random.uniform(0, 1, n)
        outcomes = np.random.binomial(1, risks, n)

        rows = [{"risk": float(risks[i]), "outcome": bool(outcomes[i]), "stratum": 0} for i in range(n)]

        result = compute_optimal_screening_curve(
            rows=rows,
            outcome_col="outcome",
            strata_features=["stratum"],
            beta=params["beta"],
            alpha_quantiles=[0.0],
            use_custom_risk_col="risk",
        )

        band_info = result["band_info"][0]

        # When α=0: Band 1 = β, Band 2 ≈ 0, Band 3 ≈ 0, Band 4 = 1-β
        # Use params tolerance for finite sample effects
        tolerance = params["tolerance"]
        assert abs(band_info["band1_pop_prop"] - params["beta"]) < tolerance
        assert band_info["band2_pop_prop"] < tolerance
        assert band_info["band3_pop_prop"] < tolerance
        assert abs(band_info["band4_pop_prop"] - (1 - params["beta"])) < tolerance

        print("\n✓ Edge case α=0 verified")
        print(f"  Band 1: {band_info['band1_pop_prop']:.4f} (expected: {params['beta']:.4f})")
        print(f"  Band 4: {band_info['band4_pop_prop']:.4f} (expected: {1 - params['beta']:.4f})")
