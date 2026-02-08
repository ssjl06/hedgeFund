"""
NVIDIA cvar_basic.ipynb & efficient_frontier.ipynb 결과 재현 및 비교 테스트

NVIDIA 노트북과 동일한 392 S&P 500 종목, 동일 파라미터 사용.
절대 수치 비교를 통한 구현 검증.
"""
import time
import numpy as np
import pandas as pd

from optimizer import (
    fetch_prices, AdvancedOptimizationParams,
    optimize_cvar_advanced, run_advanced_optimization,
    _compute_sortino, _compute_max_drawdown,
)
from scenarios import ReturnType, ScenarioMethod, compute_returns, generate_scenarios
from frontier import FrontierParams, run_frontier_analysis

SP500_TICKERS = [
    'A', 'AAPL', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN',
    'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMT', 'AMZN', 'AON', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXON', 'AXP', 'AZO',
    'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BRO', 'BSX', 'BWA', 'BXP',
    'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMI', 'CMS',
    'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CVS', 'CVX',
    'D', 'DD', 'DE', 'DECK', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG',
    'EW', 'EXC', 'EXPD', 'EXR', 'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FRT',
    'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
    'HAL', 'HAS', 'HBAN', 'HD', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
    'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
    'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR',
    'L', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS',
    'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',
    'MOS', 'MPWR', 'MRK', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
    'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
    'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OXY',
    'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU',
    'PSA', 'PTC', 'PWR', 'QCOM',
    'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
    'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYK',
    'SYY', 'T', 'TAP', 'TDY', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 'TXN',
    'TXT', 'TYL', 'UDR', 'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
    'VLO', 'VMC', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN',
    'XEL', 'XOM', 'YUM', 'ZBH', 'ZBRA',
]

# NVIDIA cvar_basic reference values (CPU CLARABEL solver)
NVIDIA_CVAR_BASIC = {
    "expected_return": 0.002529,  # daily
    "cvar": 0.025748,  # daily
    "top_weights": {
        "LLY": 0.344, "NVDA": 0.139, "MCK": 0.115, "IRM": 0.113,
        "JBL": 0.099, "COP": 0.096, "IT": 0.075, "PWR": 0.066,
        "NUE": 0.049, "FICO": 0.042, "STLD": 0.041, "XOM": 0.022,
    },
    "short_weights": {"MTCH": -0.235, "ILMN": -0.155},
    "cash": 0.200,
}

# NVIDIA efficient_frontier reference (row 16, max Sharpe)
NVIDIA_EF_TANGENT = {
    "return": 0.001484,  # daily
    "cvar": 0.021010,  # daily
    "sharpe": 2.298762,
    "risk_aversion_raw": 0.062102,
}

NVIDIA_EF_RANGE = {
    "min_return": 0.000242,
    "max_return": 0.001854,
    "min_cvar": 0.014890,
    "max_cvar": 0.037276,
}


def test_cvar_basic():
    """Test 1: Replicate NVIDIA cvar_basic.ipynb with 392 S&P 500 stocks."""
    print("=" * 70)
    print("TEST 1: CVaR Basic — 392 S&P 500 종목 (NVIDIA cvar_basic.ipynb 재현)")
    print("=" * 70)

    # 1. Download data
    print("\n[1/5] 데이터 다운로드 중 (392 종목, 2021-01-01 ~ 2024-01-01)...")
    t0 = time.time()
    prices = fetch_prices(SP500_TICKERS, "2021-01-01", "2024-01-01")
    t_dl = time.time() - t0
    valid_tickers = list(prices.columns)
    print(f"  다운로드 완료: {len(valid_tickers)}개 종목, {len(prices)}일, {t_dl:.1f}초")

    # 2. Compute log returns
    print("\n[2/5] 로그 수익률 계산...")
    returns = compute_returns(prices, ReturnType.LOG)
    print(f"  수익률 데이터: {returns.shape[0]}일 x {returns.shape[1]}종목")

    # 3. Generate KDE scenarios
    print("\n[3/5] KDE 시나리오 생성 (10,000개, bandwidth=0.01)...")
    t0 = time.time()
    scenarios = generate_scenarios(
        returns, ScenarioMethod.KDE,
        num_scenarios=10000, bandwidth=0.01, kernel="gaussian",
    )
    t_kde = time.time() - t0
    print(f"  시나리오: {scenarios.shape}, {t_kde:.1f}초")

    mean_ret = returns.mean().values

    # 4. Optimize — same params as NVIDIA
    # NVIDIA: risk_aversion=1 → in their formulation: max mu'w - 1*CVaR
    # Our formulation: min -lam*mu'w + (1-lam)*CVaR
    # When lam=0.5: min -0.5*mu'w + 0.5*CVaR ∝ min -mu'w + CVaR → same as NVIDIA risk_aversion=1
    print("\n[4/5] CVaR 최적화 (leverage=1.6, cash=[0,0.2], NVDA 10-60%, others [-0.3, 0.4])...")
    params = AdvancedOptimizationParams(
        tickers=valid_tickers,
        start_date="2021-01-01",
        end_date="2024-01-01",
        confidence_level=0.95,
        risk_aversion=0.5,  # maps to NVIDIA risk_aversion=1
        return_type="log",
        scenario_method="kde",
        num_scenarios=10000,
        kde_bandwidth=0.01,
        kde_kernel="gaussian",
        min_weight=-0.3,
        max_weight=0.4,
        asset_bounds=[{"ticker": "NVDA", "min": 0.1, "max": 0.6}],
        cash_min=0.0,
        cash_max=0.2,
        leverage_target=1.6,
        risk_free_rate=0.0,  # NVIDIA uses 0 for backtest
    )

    t0 = time.time()
    weights, cash, cvar = optimize_cvar_advanced(scenarios, mean_ret, params)
    t_opt = time.time() - t0

    exp_ret = float(mean_ret @ weights)

    print(f"  최적화 완료: {t_opt:.1f}초")
    print(f"\n{'='*50}")
    print(f"{'항목':<25} {'우리 결과':>15} {'NVIDIA':>15} {'차이':>10}")
    print(f"{'='*50}")
    print(f"{'일별 기대수익률':<25} {exp_ret:.6f} {NVIDIA_CVAR_BASIC['expected_return']:.6f} {abs(exp_ret - NVIDIA_CVAR_BASIC['expected_return']):.6f}")
    print(f"{'CVaR (95%)':<25} {cvar:.6f} {NVIDIA_CVAR_BASIC['cvar']:.6f} {abs(cvar - NVIDIA_CVAR_BASIC['cvar']):.6f}")
    print(f"{'현금 비중':<25} {cash:.4f} {NVIDIA_CVAR_BASIC['cash']:.4f} {abs(cash - NVIDIA_CVAR_BASIC['cash']):.4f}")

    # Top weights comparison
    weight_dict = {t: float(w) for t, w in zip(valid_tickers, weights)}
    sorted_weights = sorted(weight_dict.items(), key=lambda x: -x[1])

    print(f"\n--- 상위 롱 포지션 비교 ---")
    print(f"{'종목':<8} {'우리':>10} {'NVIDIA':>10} {'차이':>10}")
    our_top_long = [(t, w) for t, w in sorted_weights if w > 0.01][:15]
    for t, w in our_top_long:
        nv = NVIDIA_CVAR_BASIC["top_weights"].get(t, 0)
        diff = abs(w - nv)
        match = "✓" if diff < 0.05 else "△" if diff < 0.15 else "✗"
        print(f"  {t:<6} {w:>+10.4f} {nv:>+10.4f} {diff:>10.4f} {match}")

    print(f"\n--- 숏 포지션 비교 ---")
    our_shorts = [(t, w) for t, w in sorted_weights if w < -0.01]
    for t, w in our_shorts[:5]:
        nv = NVIDIA_CVAR_BASIC["short_weights"].get(t, 0)
        diff = abs(w - nv)
        match = "✓" if diff < 0.05 else "△" if diff < 0.15 else "✗"
        print(f"  {t:<6} {w:>+10.4f} {nv:>+10.4f} {diff:>10.4f} {match}")

    total_long = sum(w for w in weights if w > 0)
    total_short = sum(w for w in weights if w < 0)
    print(f"\n  총 롱: {total_long:.4f} (NVIDIA: ~1.200)")
    print(f"  총 숏: {total_short:.4f} (NVIDIA: ~-0.390)")
    print(f"  현금: {cash:.4f} (NVIDIA: 0.200)")
    print(f"  순노출: {total_long + total_short + cash:.4f}")

    # Qualitative check
    nvidia_top = set(NVIDIA_CVAR_BASIC["top_weights"].keys())
    our_top = set(t for t, _ in our_top_long)
    overlap = nvidia_top & our_top
    print(f"\n  NVIDIA 상위 12 종목 중 우리도 선택한 종목: {len(overlap)}/{len(nvidia_top)}")
    print(f"  일치 종목: {sorted(overlap)}")
    print(f"  NVIDIA에만: {sorted(nvidia_top - our_top)}")
    print(f"  우리에만: {sorted(our_top - nvidia_top)[:10]}")

    return True


def test_efficient_frontier():
    """Test 2: Replicate NVIDIA efficient_frontier.ipynb."""
    print("\n\n" + "=" * 70)
    print("TEST 2: 효율적 프론티어 — 392 S&P 500 종목 (NVIDIA efficient_frontier.ipynb 재현)")
    print("=" * 70)

    # NVIDIA uses: long-only, no cash, leverage=1, confidence=0.95
    # Log returns, KDE scenarios (10000, bw=0.01), 30 log-spaced lambdas from 10^-3 to 10^1
    # Date range: 2022-01-01 to 2024-07-01
    # Custom portfolio: AAPL 30%, LLY 20%, MSFT 50%

    print("\n[1/4] 데이터 다운로드 중 (392 종목, 2022-01-01 ~ 2024-07-01)...")
    t0 = time.time()
    prices = fetch_prices(SP500_TICKERS, "2022-01-01", "2024-07-01")
    t_dl = time.time() - t0
    valid_tickers = list(prices.columns)
    print(f"  {len(valid_tickers)}개 종목, {len(prices)}일, {t_dl:.1f}초")

    print("\n[2/4] 로그 수익률 + KDE 시나리오 생성...")
    returns = compute_returns(prices, ReturnType.LOG)
    t0 = time.time()
    scenarios = generate_scenarios(
        returns, ScenarioMethod.KDE,
        num_scenarios=10000, bandwidth=0.01, kernel="gaussian",
    )
    t_kde = time.time() - t0
    print(f"  수익률: {returns.shape}, 시나리오: {scenarios.shape}, {t_kde:.1f}초")

    mean_ret = returns.mean().values

    # 3. Frontier sweep — same as NVIDIA: 30 points, 10^-3 to 10^1
    print("\n[3/4] 로그 스케일 프론티어 스윕 (30 포인트)...")
    # We use historical scenarios instead of KDE for the frontier to keep it tractable
    # Actually let's use the same KDE scenarios
    from optimizer import OptimizationParams, optimize_cvar

    raw_lambdas = np.logspace(-3, 1, 30)
    frontier_points = []

    t0 = time.time()
    for i, raw_lam in enumerate(raw_lambdas):
        lam_norm = raw_lam / (1.0 + raw_lam)
        lam_norm = max(0.01, min(0.99, lam_norm))

        p = OptimizationParams(
            tickers=valid_tickers,
            start_date="2022-01-01",
            end_date="2024-07-01",
            confidence_level=0.95,
            risk_aversion=lam_norm,
            min_weight=0.0,
            max_weight=1.0,
        )
        try:
            w, cvar_val = optimize_cvar(scenarios, mean_ret, p)
            port_ret = float(mean_ret @ w)
            port_vol = float(np.std(scenarios @ w, ddof=1))
            sharpe = port_ret / port_vol if port_vol > 0 else 0
            frontier_points.append({
                "idx": i,
                "lambda_raw": float(raw_lam),
                "lambda_norm": float(lam_norm),
                "return": port_ret,
                "cvar": cvar_val,
                "volatility": port_vol,
                "sharpe": sharpe,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/30] λ={raw_lam:.4f}, ret={port_ret:.6f}, cvar={cvar_val:.6f}")
        except RuntimeError as e:
            print(f"  [{i+1}/30] λ={raw_lam:.4f} FAILED: {e}")

    t_frontier = time.time() - t0
    print(f"  프론티어 완료: {len(frontier_points)}개 포인트, {t_frontier:.1f}초")

    # Compare with NVIDIA results
    print(f"\n{'='*60}")
    print("NVIDIA 효율적 프론티어 결과 비교")
    print(f"{'='*60}")

    if frontier_points:
        rets = [p["return"] for p in frontier_points]
        cvars = [p["cvar"] for p in frontier_points]
        sharpes = [p["sharpe"] for p in frontier_points]

        best_sharpe_idx = np.argmax(sharpes)
        best = frontier_points[best_sharpe_idx]

        print(f"\n{'항목':<30} {'우리':>15} {'NVIDIA':>15}")
        print(f"{'-'*60}")
        print(f"{'프론티어 포인트 수':<30} {len(frontier_points):>15} {'30':>15}")
        print(f"{'최소 일별 수익률':<30} {min(rets):>15.6f} {NVIDIA_EF_RANGE['min_return']:>15.6f}")
        print(f"{'최대 일별 수익률':<30} {max(rets):>15.6f} {NVIDIA_EF_RANGE['max_return']:>15.6f}")
        print(f"{'최소 CVaR':<30} {min(cvars):>15.6f} {NVIDIA_EF_RANGE['min_cvar']:>15.6f}")
        print(f"{'최대 CVaR':<30} {max(cvars):>15.6f} {NVIDIA_EF_RANGE['max_cvar']:>15.6f}")

        print(f"\n--- 접선 포트폴리오 (최대 Sharpe) ---")
        print(f"{'항목':<30} {'우리':>15} {'NVIDIA':>15}")
        print(f"{'-'*60}")
        print(f"{'λ (raw)':<30} {best['lambda_raw']:>15.6f} {NVIDIA_EF_TANGENT['risk_aversion_raw']:>15.6f}")
        print(f"{'일별 수익률':<30} {best['return']:>15.6f} {NVIDIA_EF_TANGENT['return']:>15.6f}")
        print(f"{'CVaR':<30} {best['cvar']:>15.6f} {NVIDIA_EF_TANGENT['cvar']:>15.6f}")
        print(f"{'Sharpe (daily)':<30} {best['sharpe']:>15.6f} {NVIDIA_EF_TANGENT['sharpe']:>15.6f}")

        # Show first and last 5 points
        print(f"\n--- 프론티어 전체 (처음 5개) ---")
        print(f"{'λ':>12} {'return':>12} {'cvar':>12} {'vol':>12} {'sharpe':>10}")
        for p in frontier_points[:5]:
            print(f"{p['lambda_raw']:>12.6f} {p['return']:>12.6f} {p['cvar']:>12.6f} {p['volatility']:>12.6f} {p['sharpe']:>10.4f}")
        print(f"  ...")
        for p in frontier_points[-5:]:
            print(f"{p['lambda_raw']:>12.6f} {p['return']:>12.6f} {p['cvar']:>12.6f} {p['volatility']:>12.6f} {p['sharpe']:>10.4f}")

    # 4. Custom portfolio evaluation
    print(f"\n[4/4] 비교 포트폴리오 평가 (AAPL 30%, LLY 20%, MSFT 50%)...")
    custom_w = np.zeros(len(valid_tickers))
    for i, t in enumerate(valid_tickers):
        if t == "AAPL":
            custom_w[i] = 0.3
        elif t == "LLY":
            custom_w[i] = 0.2
        elif t == "MSFT":
            custom_w[i] = 0.5

    custom_ret = float(mean_ret @ custom_w)
    custom_vol = float(np.std(scenarios @ custom_w, ddof=1))
    custom_sharpe = custom_ret / custom_vol if custom_vol > 0 else 0

    # CVaR of custom portfolio
    beta = 0.95
    custom_scen = scenarios @ custom_w
    var_thresh = np.percentile(custom_scen, (1 - beta) * 100)
    tail = custom_scen[custom_scen <= var_thresh]
    custom_cvar = float(tail.mean()) if len(tail) > 0 else float(var_thresh)

    print(f"  일별 수익률: {custom_ret:.6f}")
    print(f"  CVaR: {custom_cvar:.6f}")
    print(f"  Sharpe (daily): {custom_sharpe:.4f}")

    return True


if __name__ == "__main__":
    print("NVIDIA 노트북 대비 절대 수치 비교 테스트")
    print("=" * 70)
    print(f"종목 수: {len(SP500_TICKERS)}")
    print()

    try:
        test_cvar_basic()
    except Exception as e:
        print(f"\nTEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_efficient_frontier()
    except Exception as e:
        print(f"\nTEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)
    print("""
비교 시 유의사항:
1. 시나리오 차이: KDE 시드가 다르므로 10,000개 시나리오가 정확히 같지 않음
2. 솔버 차이: NVIDIA는 cuOpt(PDLP), 우리는 scipy(HiGHS) — 수치적 차이 불가피
3. 데이터 차이: yfinance 다운로드 시점/보정 차이로 주가 데이터가 약간 다를 수 있음
4. 동일 구조 확인: 종목 선택 패턴, 롱/숏 비율, 현금 사용, CVaR 범위가 유사하면 정상
""")
