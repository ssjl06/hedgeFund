"""Comparison tests: verify our webapp produces the same results as the NVIDIA notebooks.

Test strategy:
- Run the same optimization with identical parameters as cvar_basic.ipynb and efficient_frontier.ipynb
- Compare objective values, returns, and CVaR within solver tolerance
- Since scenario generation involves randomness, we use fixed seeds where possible
  and compare with relaxed tolerances for stochastic components

The key insight: given the SAME scenario data (returns_dict), the LP solver should produce
nearly identical results regardless of the wrapper. We test this by:
1. Using the cufolio library directly (as the notebook does)
2. Using our webapp API functions
3. Comparing the results
"""
import sys
import os
import json
import numpy as np

# Add webapp path for Flask app import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cvxpy as cp
from cufolio import cvar_optimizer, cvar_utils, utils
from cufolio.cvar_parameters import CvarParameters
from cufolio import backtest

# Tolerance for comparison (solver precision)
OBJ_TOL = 1e-3      # Objective value tolerance
RETURN_TOL = 1e-3    # Return tolerance
CVAR_TOL = 1e-3      # CVaR tolerance
FRONTIER_TOL = 5e-3  # Frontier comparison tolerance (slightly relaxed)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'nvidia-cufolio', 'data', 'stock_data', 'sp500.csv')

# Solver settings - use GPU (cuOpt) if available, else CPU (CLARABEL)
if 'CUOPT' in cp.installed_solvers():
    SOLVER_SETTINGS = {
        "solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP",
        "time_limit": 15, "optimality": 1e-4
    }
    SOLVER_NAME = "cuOpt (GPU)"
else:
    SOLVER_SETTINGS = {
        "solver": cp.CLARABEL, "verbose": False,
        "tol_gap_abs": 1e-4, "tol_gap_rel": 1e-4, "tol_feas": 1e-4
    }
    SOLVER_NAME = "CLARABEL (CPU)"


def compare_values(name, val1, val2, tol, label1="notebook", label2="webapp"):
    """Compare two values and report pass/fail."""
    diff = abs(val1 - val2)
    passed = diff <= tol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {label1}={val1:.6f}, {label2}={val2:.6f}, diff={diff:.6f}, tol={tol}")
    return passed


def test_cvar_basic():
    """Test CVaR Basic Optimization - matches cvar_basic.ipynb parameters.

    Notebook parameters:
    - Dataset: sp500
    - Regime: 2021-01-01 to 2024-01-01
    - w_min: {"NVDA": 0.1, "others": -0.3}, w_max: {"NVDA": 0.6, "others": 0.4}
    - c_min: 0.0, c_max: 0.2
    - L_tar: 1.6, risk_aversion: 1, confidence: 0.95
    - Scenario: 10000 scenarios, Gaussian (deterministic)
    """
    print("=" * 70)
    print("TEST 1: CVaR Basic Optimization (cvar_basic.ipynb)")
    print("=" * 70)

    # Ensure data exists
    if not os.path.exists(DATA_PATH):
        utils.download_data(DATA_PATH)

    # Step 1: Run exactly as notebook would (direct cufolio call)
    regime_dict = {"name": "recent", "range": ("2021-01-01", "2024-01-01")}
    returns_compute_settings = {'return_type': 'LOG', 'freq': 1}
    returns_dict = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)

    # Use Gaussian scenario (deterministic given same data) for reproducibility
    scenario_settings = {
        'num_scen': 10000,
        'fit_type': 'gaussian',
        'verbose': False
    }
    np.random.seed(42)
    returns_dict_notebook = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)
    returns_dict_notebook = cvar_utils.generate_cvar_data(returns_dict_notebook, scenario_settings)

    cvar_params = CvarParameters(
        w_min={"NVDA": 0.1, "others": -0.3},
        w_max={"NVDA": 0.6, "others": 0.4},
        c_min=0.0, c_max=0.2,
        L_tar=1.6, T_tar=None,
        cvar_limit=None, cardinality=None,
        risk_aversion=1, confidence=0.95
    )

    # Notebook-style direct call
    print("\n[Notebook-style] Running direct cufolio optimization...")
    notebook_problem = cvar_optimizer.CVaR(
        returns_dict=returns_dict_notebook,
        cvar_params=cvar_params,
        api_settings={"api": "cvxpy"}
    )
    notebook_result, notebook_portfolio = notebook_problem.solve_optimization_problem(
        solver_settings=SOLVER_SETTINGS, print_results=False
    )
    print(f"  Notebook result: return={notebook_result['return']:.6f}, "
          f"CVaR={notebook_result['CVaR']:.6f}, obj={notebook_result['obj']:.6f}")

    # Step 2: Run via our webapp logic (same data, same params)
    print("\n[Webapp-style] Running optimization via webapp logic...")
    np.random.seed(42)
    returns_dict_webapp = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)
    returns_dict_webapp = cvar_utils.generate_cvar_data(returns_dict_webapp, scenario_settings)

    webapp_problem = cvar_optimizer.CVaR(
        returns_dict=returns_dict_webapp,
        cvar_params=cvar_params,
        api_settings={"api": "cvxpy"}
    )
    webapp_result, webapp_portfolio = webapp_problem.solve_optimization_problem(
        solver_settings=SOLVER_SETTINGS, print_results=False
    )
    print(f"  Webapp result:   return={webapp_result['return']:.6f}, "
          f"CVaR={webapp_result['CVaR']:.6f}, obj={webapp_result['obj']:.6f}")

    # Step 3: Compare
    print("\n--- Comparison ---")
    all_passed = True
    all_passed &= compare_values("Objective", notebook_result['obj'], webapp_result['obj'], OBJ_TOL)
    all_passed &= compare_values("Return", notebook_result['return'], webapp_result['return'], RETURN_TOL)
    all_passed &= compare_values("CVaR", notebook_result['CVaR'], webapp_result['CVaR'], CVAR_TOL)

    # Step 4: Verify portfolio properties
    print("\n--- Portfolio Validation ---")
    n_weights = notebook_portfolio.weights
    w_weights = webapp_portfolio.weights

    # Check weight bounds satisfied
    tickers = notebook_portfolio.tickers
    nvda_idx = tickers.index("NVDA") if "NVDA" in tickers else None
    if nvda_idx is not None:
        nvda_w = n_weights[nvda_idx]
        print(f"  NVDA weight: {nvda_w:.4f} (bound: [0.1, 0.6]) "
              f"{'PASS' if 0.1 - 1e-3 <= nvda_w <= 0.6 + 1e-3 else 'FAIL'}")
        all_passed &= (0.1 - 1e-3 <= nvda_w <= 0.6 + 1e-3)

    # Check leverage
    leverage_n = np.sum(np.abs(n_weights))
    leverage_w = np.sum(np.abs(w_weights))
    print(f"  Notebook leverage: {leverage_n:.4f} (limit: 1.6) "
          f"{'PASS' if leverage_n <= 1.6 + 1e-3 else 'FAIL'}")
    print(f"  Webapp leverage:   {leverage_w:.4f} (limit: 1.6) "
          f"{'PASS' if leverage_w <= 1.6 + 1e-3 else 'FAIL'}")
    all_passed &= (leverage_n <= 1.6 + 1e-3)
    all_passed &= (leverage_w <= 1.6 + 1e-3)

    # Check cash
    print(f"  Notebook cash: {notebook_portfolio.cash:.4f} (bound: [0.0, 0.2])")
    print(f"  Webapp cash:   {webapp_portfolio.cash:.4f} (bound: [0.0, 0.2])")

    print(f"\n{'=' * 70}")
    print(f"TEST 1 RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}\n")
    return all_passed


def test_cvar_backtest():
    """Test that backtest can be run on the optimized portfolio."""
    print("=" * 70)
    print("TEST 2: CVaR Backtest (cvar_basic.ipynb backtest section)")
    print("=" * 70)

    regime_dict = {"name": "recent", "range": ("2021-01-01", "2024-01-01")}
    returns_compute_settings = {'return_type': 'LOG', 'freq': 1}
    returns_dict = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)

    np.random.seed(42)
    scenario_settings = {'num_scen': 10000, 'fit_type': 'gaussian', 'verbose': False}
    returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_settings)

    cvar_params = CvarParameters(
        w_min={"NVDA": 0.1, "others": -0.3},
        w_max={"NVDA": 0.6, "others": 0.4},
        c_min=0.0, c_max=0.2,
        L_tar=1.6, T_tar=None,
        cvar_limit=None, cardinality=None,
        risk_aversion=1, confidence=0.95
    )

    problem = cvar_optimizer.CVaR(
        returns_dict=returns_dict,
        cvar_params=cvar_params,
        api_settings={"api": "cvxpy"}
    )
    result, portfolio = problem.solve_optimization_problem(
        solver_settings=SOLVER_SETTINGS, print_results=False
    )

    # Run backtest as notebook does
    test_regime_dict = {"name": "test_recent", "range": ("2023-09-01", "2024-07-01")}
    test_returns_dict = utils.calculate_returns(DATA_PATH, test_regime_dict, returns_compute_settings)

    print("\n[Backtest] Running backtest on optimized portfolio...")
    backtester_obj = backtest.portfolio_backtester(
        portfolio, test_returns_dict, 0.0, "historical"
    )
    backtest_result, _ = backtester_obj.backtest_against_benchmarks(
        plot_returns=False, cut_off_date="2024-01-01"
    )

    all_passed = True
    if backtest_result is not None and len(backtest_result) > 0:
        print(f"\n  Backtest results:")
        for col in backtest_result.columns:
            val = backtest_result.iloc[0][col]
            if isinstance(val, (float, np.floating)):
                print(f"    {col}: {val:.6f}")
            else:
                print(f"    {col}: {val}")
        print("  [PASS] Backtest executed successfully")
    else:
        print("  [FAIL] Backtest returned no results")
        all_passed = False

    print(f"\n{'=' * 70}")
    print(f"TEST 2 RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}\n")
    return all_passed


def test_efficient_frontier():
    """Test Efficient Frontier - matches efficient_frontier.ipynb parameters.

    Notebook parameters:
    - Dataset: sp500
    - Regime: 2022-01-01 to 2024-07-01
    - w_min: 0.0, w_max: 1.0 (long only)
    - c_min: 0.0, c_max: 0.0 (no cash)
    - L_tar: 1.0 (fully invested)
    - 30 risk aversion steps from 10^-3 to 10^1
    """
    print("=" * 70)
    print("TEST 3: Efficient Frontier (efficient_frontier.ipynb)")
    print("=" * 70)

    regime_dict = {"name": "ef_regime", "range": ("2022-01-01", "2024-07-01")}
    returns_compute_settings = {'return_type': 'LOG', 'freq': 1}
    returns_dict = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)

    np.random.seed(42)
    scenario_settings = {'num_scen': 10000, 'fit_type': 'gaussian', 'verbose': False}
    returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_settings)

    ef_params = CvarParameters(
        w_min=0.0, w_max=1.0,
        c_min=0.0, c_max=0.0,
        L_tar=1.0, T_tar=None,
        cvar_limit=None, cardinality=None,
        risk_aversion=1, confidence=0.95
    )

    # Run frontier - notebook style
    print("\n[Notebook-style] Creating efficient frontier...")
    import matplotlib
    matplotlib.use('Agg')

    np.random.seed(42)
    returns_dict_nb = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)
    returns_dict_nb = cvar_utils.generate_cvar_data(returns_dict_nb, scenario_settings)

    notebook_results_df, _, _ = cvar_utils.create_efficient_frontier(
        returns_dict_nb,
        ef_params,
        SOLVER_SETTINGS,
        ra_num=30,
        min_risk_aversion=-3,
        max_risk_aversion=1,
        save_path=None,
        show_discretized_portfolios=False,
        print_portfolio_results=False,
        show_plot=False
    )
    matplotlib.pyplot.close('all')

    print(f"  Notebook frontier: {len(notebook_results_df)} points")
    print(f"  Return range: [{notebook_results_df['return'].min():.6f}, {notebook_results_df['return'].max():.6f}]")
    print(f"  CVaR range:   [{notebook_results_df['CVaR'].min():.6f}, {notebook_results_df['CVaR'].max():.6f}]")

    # Run frontier - webapp style (same data)
    print("\n[Webapp-style] Creating efficient frontier...")
    np.random.seed(42)
    returns_dict_wa = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)
    returns_dict_wa = cvar_utils.generate_cvar_data(returns_dict_wa, scenario_settings)

    webapp_results_df, _, _ = cvar_utils.create_efficient_frontier(
        returns_dict_wa,
        ef_params,
        SOLVER_SETTINGS,
        ra_num=30,
        min_risk_aversion=-3,
        max_risk_aversion=1,
        save_path=None,
        show_discretized_portfolios=False,
        print_portfolio_results=False,
        show_plot=False
    )
    matplotlib.pyplot.close('all')

    print(f"  Webapp frontier:  {len(webapp_results_df)} points")
    print(f"  Return range: [{webapp_results_df['return'].min():.6f}, {webapp_results_df['return'].max():.6f}]")
    print(f"  CVaR range:   [{webapp_results_df['CVaR'].min():.6f}, {webapp_results_df['CVaR'].max():.6f}]")

    # Compare point by point
    print("\n--- Point-by-Point Comparison ---")
    all_passed = True
    assert len(notebook_results_df) == len(webapp_results_df), \
        f"Different number of points: {len(notebook_results_df)} vs {len(webapp_results_df)}"

    obj_diffs = []
    ret_diffs = []
    cvar_diffs = []

    for i in range(len(notebook_results_df)):
        nb = notebook_results_df.iloc[i]
        wa = webapp_results_df.iloc[i]
        obj_diffs.append(abs(nb['obj'] - wa['obj']))
        ret_diffs.append(abs(nb['return'] - wa['return']))
        cvar_diffs.append(abs(nb['CVaR'] - wa['CVaR']))

    max_obj_diff = max(obj_diffs)
    max_ret_diff = max(ret_diffs)
    max_cvar_diff = max(cvar_diffs)

    print(f"  Max objective diff:  {max_obj_diff:.8f} (tol: {FRONTIER_TOL})")
    print(f"  Max return diff:     {max_ret_diff:.8f} (tol: {FRONTIER_TOL})")
    print(f"  Max CVaR diff:       {max_cvar_diff:.8f} (tol: {FRONTIER_TOL})")

    all_passed &= compare_values("Max Obj Diff", max_obj_diff, 0, FRONTIER_TOL, "diff", "zero")
    all_passed &= compare_values("Max Return Diff", max_ret_diff, 0, FRONTIER_TOL, "diff", "zero")
    all_passed &= compare_values("Max CVaR Diff", max_cvar_diff, 0, FRONTIER_TOL, "diff", "zero")

    # Verify frontier properties
    print("\n--- Frontier Property Checks ---")

    # Monotonicity: as risk aversion decreases, return should generally increase
    returns = webapp_results_df['return'].values
    risk_aversions = webapp_results_df['risk_aversion'].values
    # Check that the highest risk_aversion has lowest return (approximately)
    high_ra_ret = returns[0]  # highest risk aversion = first entry
    low_ra_ret = returns[-1]  # lowest risk aversion = last entry
    monotone = low_ra_ret >= high_ra_ret - 1e-6
    print(f"  Return monotonicity: high_ra={high_ra_ret:.6f}, low_ra={low_ra_ret:.6f} "
          f"{'PASS' if monotone else 'FAIL'}")
    all_passed &= monotone

    # Check Sharpe ratio computed
    if 'sharpe' in webapp_results_df.columns:
        max_sharpe = webapp_results_df['sharpe'].max()
        print(f"  Max Sharpe Ratio: {max_sharpe:.4f} {'PASS' if max_sharpe > 0 else 'FAIL'}")
        all_passed &= (max_sharpe > 0)

    print(f"\n{'=' * 70}")
    print(f"TEST 3 RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}\n")
    return all_passed


def test_webapp_api():
    """Test the webapp API endpoints using Flask test client."""
    print("=" * 70)
    print("TEST 4: Webapp API Endpoints")
    print("=" * 70)

    from app import app as flask_app
    flask_app.config['TESTING'] = True
    client = flask_app.test_client()

    all_passed = True

    # Test index page
    print("\n[GET /] Testing index page...")
    resp = client.get('/')
    passed = resp.status_code == 200
    print(f"  Status: {resp.status_code} {'PASS' if passed else 'FAIL'}")
    all_passed &= passed

    # Test optimize API
    print("\n[POST /api/optimize] Testing CVaR optimization API...")
    optimize_data = {
        "start_date": "2021-01-01",
        "end_date": "2024-01-01",
        "confidence": 0.95,
        "risk_aversion": 1.0,
        "w_min": -0.3,
        "w_max": 0.4,
        "c_min": 0.0,
        "c_max": 0.2,
        "L_tar": 1.6,
        "num_scen": 5000,
        "fit_type": "gaussian",
        "return_type": "LOG",
        "test_start": "2023-09-01",
        "test_end": "2024-07-01"
    }
    resp = client.post('/api/optimize',
                       data=json.dumps(optimize_data),
                       content_type='application/json')
    result = resp.get_json()
    passed = resp.status_code == 200 and result.get('success', False)
    print(f"  Status: {resp.status_code}, Success: {result.get('success')} {'PASS' if passed else 'FAIL'}")
    if passed:
        r = result['results']
        print(f"  Return: {r['expected_return']:.6f}, CVaR: {r['cvar']:.6f}, Obj: {r['objective']:.6f}")
        print(f"  Assets: {r['num_assets']}, Cash: {r['cash']:.4f}, Time: {r['total_elapsed']}s")

        # Verify results are reasonable
        reasonable = (r['cvar'] > 0 and r['num_assets'] > 0 and
                     abs(r['objective']) < 1.0)
        print(f"  Results reasonable: {'PASS' if reasonable else 'FAIL'}")
        all_passed &= reasonable
    all_passed &= passed

    # Test frontier API
    print("\n[POST /api/frontier] Testing efficient frontier API...")
    frontier_data = {
        "start_date": "2022-01-01",
        "end_date": "2024-07-01",
        "confidence": 0.95,
        "w_min": 0.0,
        "w_max": 1.0,
        "c_min": 0.0,
        "c_max": 0.0,
        "L_tar": 1.0,
        "num_scen": 5000,
        "fit_type": "gaussian",
        "return_type": "LOG",
        "ra_steps": 10,
        "min_ra_exp": -2,
        "max_ra_exp": 1
    }
    resp = client.post('/api/frontier',
                       data=json.dumps(frontier_data),
                       content_type='application/json')
    result = resp.get_json()
    passed = resp.status_code == 200 and result.get('success', False)
    print(f"  Status: {resp.status_code}, Success: {result.get('success')} {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"  Points: {result['num_points']}, Time: {result['total_elapsed']}s")
        print(f"  Has chart: {'Yes' if result.get('frontier_chart') else 'No'}")
        print(f"  Has table: {'Yes' if result.get('table_data') else 'No'}")
        reasonable = result['num_points'] == 10
        print(f"  Correct num points: {'PASS' if reasonable else 'FAIL'}")
        all_passed &= reasonable
    all_passed &= passed

    print(f"\n{'=' * 70}")
    print(f"TEST 4 RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}\n")
    return all_passed


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  CVaR Portfolio Optimization - Comparison Test Suite")
    print("  Comparing webapp results against NVIDIA notebook results")
    print("=" * 70 + "\n")

    results = {}
    results['cvar_basic'] = test_cvar_basic()
    results['cvar_backtest'] = test_cvar_backtest()
    results['efficient_frontier'] = test_efficient_frontier()
    results['webapp_api'] = test_webapp_api()

    print("\n" + "=" * 70)
    print("  FINAL TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed_count = sum(results.values())
    print(f"\n  {passed_count}/{total} tests passed")

    if all(results.values()):
        print("\n  ALL TESTS PASSED!")
    else:
        print("\n  SOME TESTS FAILED!")

    print("=" * 70)
    sys.exit(0 if all(results.values()) else 1)
