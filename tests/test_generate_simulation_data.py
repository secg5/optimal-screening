"""Unit tests for generate_simulation_data."""

from __future__ import annotations

import numpy as np

from optimal_screening.analysis import generate_simulation_data


class TestGenerateSimulationData:
    """Tests for the Beta-Binomial simulation helper."""

    def test_uniform_mean_risk(self):
        """Beta(1,1) = Uniform(0,1): mean risk should be ~0.5."""
        risk_scores, _ = generate_simulation_data(a=1.0, b=1.0, size=100_000, seed=42)
        assert abs(np.mean(risk_scores) - 0.5) < 0.01

    def test_uniform_outcome_rate(self):
        """Beta(1,1) risks fed to Binomial should yield ~50% positive outcomes."""
        _, outcomes = generate_simulation_data(a=1.0, b=1.0, size=100_000, seed=42)
        rate = np.mean(outcomes)
        assert abs(rate - 0.5) < 0.02, f"Expected ~0.50, got {rate:.4f}"

    def test_beta_1_3_outcome_rate(self):
        """Beta(1,3) has E[X]=0.25, so outcome rate should be ~25%."""
        _, outcomes = generate_simulation_data(a=1.0, b=3.0, size=100_000, seed=42)
        rate = np.mean(outcomes)
        assert abs(rate - 0.25) < 0.02, f"Expected ~0.25, got {rate:.4f}"

    def test_beta_1_3_mean_risk(self):
        """Beta(1,3) has E[X] = 1/(1+3) = 0.25."""
        risk_scores, _ = generate_simulation_data(a=1.0, b=3.0, size=100_000, seed=42)
        assert abs(np.mean(risk_scores) - 0.25) < 0.01

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        r1, o1 = generate_simulation_data(a=2.0, b=5.0, size=1_000, seed=123)
        r2, o2 = generate_simulation_data(a=2.0, b=5.0, size=1_000, seed=123)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        r1, _ = generate_simulation_data(a=2.0, b=5.0, size=1_000, seed=0)
        r2, _ = generate_simulation_data(a=2.0, b=5.0, size=1_000, seed=1)
        assert not np.array_equal(r1, r2)

    def test_output_shapes(self):
        """Returned arrays should have the requested size."""
        risk_scores, outcomes = generate_simulation_data(a=1.0, b=1.0, size=500, seed=0)
        assert risk_scores.shape == (500,)
        assert outcomes.shape == (500,)

    def test_risk_scores_in_unit_interval(self):
        """All risk scores must lie in [0, 1]."""
        risk_scores, _ = generate_simulation_data(a=0.5, b=0.5, size=10_000, seed=42)
        assert np.all(risk_scores >= 0.0)
        assert np.all(risk_scores <= 1.0)

    def test_outcomes_are_binary(self):
        """Outcomes must be 0 or 1."""
        _, outcomes = generate_simulation_data(a=2.0, b=2.0, size=10_000, seed=42)
        assert set(np.unique(outcomes)).issubset({0, 1})

    def test_high_risk_high_outcome_rate(self):
        """Beta(9,1) has E[X]=0.9, so outcome rate should be ~90%."""
        _, outcomes = generate_simulation_data(a=9.0, b=1.0, size=100_000, seed=42)
        rate = np.mean(outcomes)
        assert abs(rate - 0.9) < 0.02, f"Expected ~0.90, got {rate:.4f}"

    def test_point_mass_all_risks_equal(self):
        """Point mass at 0.5: every risk score should be exactly 0.5."""
        risk_scores, _ = generate_simulation_data(size=1_000, seed=42, point_mass=0.5)
        np.testing.assert_array_equal(risk_scores, 0.5)

    def test_point_mass_outcome_rate(self):
        """Point mass at 0.5 should yield ~50% positive outcomes."""
        _, outcomes = generate_simulation_data(size=100_000, seed=42, point_mass=0.5)
        rate = np.mean(outcomes)
        assert abs(rate - 0.5) < 0.02, f"Expected ~0.50, got {rate:.4f}"

    def test_point_mass_deterministic_risk(self):
        """Point mass at 0.25: every risk score is exactly 0.25, outcomes ~25%."""
        risk_scores, outcomes = generate_simulation_data(size=100_000, seed=42, point_mass=0.25)
        np.testing.assert_array_equal(risk_scores, 0.25)
        rate = np.mean(outcomes)
        assert abs(rate - 0.25) < 0.02, f"Expected ~0.25, got {rate:.4f}"
