#!/usr/bin/env python
"""Run tests without pytest."""

import sys

from test_optimal_screening import TestOptimalScreeningCurveUniform


def main():
    """Run all tests."""
    test_class = TestOptimalScreeningCurveUniform()

    # Create params
    params = {
        "n_samples": 10000,
        "n_simulations": 100,
        "beta": 0.6,
        "alpha_values": [0.0, 0.1, 0.2, 0.3],  # Keep alpha well below beta for stable results
        "tolerance": 0.02,
    }

    print("=" * 70)
    print("Running Optimal Screening Curve Tests with Uniform Distribution")
    print("=" * 70)

    try:
        print("\n[Test 1/4] Single alpha value test...")
        test_class.test_uniform_band_structure_single_alpha(params)
        print("✓ Test 1 PASSED")
    except AssertionError as e:
        print(f"✗ Test 1 FAILED: {e}")
        sys.exit(1)

    try:
        print("\n[Test 2/4] Multiple simulations test (100 simulations)...")
        test_class.test_uniform_distribution_multiple_simulations(params)
        print("✓ Test 2 PASSED")
    except AssertionError as e:
        print(f"✗ Test 2 FAILED: {e}")
        sys.exit(1)

    try:
        print("\n[Test 3/4] Band mass conservation test...")
        test_class.test_band_mass_conservation(params)
        print("✓ Test 3 PASSED")
    except AssertionError as e:
        print(f"✗ Test 3 FAILED: {e}")
        sys.exit(1)

    try:
        print("\n[Test 4/4] Edge case α=0 test...")
        test_class.test_alpha_zero_edge_case(params)
        print("✓ Test 4 PASSED")
    except AssertionError as e:
        print(f"✗ Test 4 FAILED: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Simulations: {params['n_simulations']}")
    print(f"  - Sample size per simulation: {params['n_samples']}")
    print(f"  - Treatment budget (β): {params['beta']:.1f}")
    print(f"  - Alpha values tested: {params['alpha_values']}")
    print(f"  - Tolerance: ±{params['tolerance']:.2%}")


if __name__ == "__main__":
    main()
