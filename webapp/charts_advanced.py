"""
Advanced chart generators for Feature 1: CVaR Portfolio Optimization.

Scenario distribution, train/test backtest, benchmark comparison,
and enhanced portfolio summary charts.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import AdvancedOptimizationResult


def _to_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


# ---------------------------------------------------------------------------
# 1. Scenario distribution: historical vs generated
# ---------------------------------------------------------------------------
def scenario_distribution_chart(result: AdvancedOptimizationResult) -> dict:
    """Histogram overlay of historical returns vs generated scenarios."""
    n_assets = len(result.params.tickers)
    # Show distribution of portfolio-level returns
    weights = np.array(list(result.weights.values()))

    hist_ret = result.historical_returns @ weights
    scen_ret = result.scenarios @ weights

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=hist_ret * 100, nbinsx=60,
        name="과거 수익률",
        marker_color="steelblue", opacity=0.6,
    ))

    if result.params.scenario_method != "historical":
        fig.add_trace(go.Histogram(
            x=scen_ret * 100, nbinsx=60,
            name=f"생성 시나리오 ({result.params.scenario_method.upper()})",
            marker_color="coral", opacity=0.5,
        ))

    fig.update_layout(
        title="포트폴리오 수익률 시나리오 분포",
        xaxis_title="일별 수익률 (%)", yaxis_title="빈도",
        barmode="overlay",
        height=420, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 2. Backtest with train/test split
# ---------------------------------------------------------------------------
def backtest_split_chart(result: AdvancedOptimizationResult) -> dict:
    """Cumulative returns with train/test vertical divider + benchmark lines."""
    bt = result.backtest
    fig = go.Figure()

    # Optimal portfolio
    fig.add_trace(go.Scatter(
        x=bt.cumulative.index, y=bt.cumulative.values,
        mode="lines", name="최적 포트폴리오",
        line=dict(color="crimson", width=2.5),
    ))

    # Benchmarks
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    if bt.cumulative_benchmarks is not None:
        for i, col in enumerate(bt.cumulative_benchmarks.columns):
            if col == "Optimal":
                continue
            fig.add_trace(go.Scatter(
                x=bt.cumulative_benchmarks.index,
                y=bt.cumulative_benchmarks[col].values,
                mode="lines", name=col,
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
            ))

    # Train/test split line
    if bt.split_date:
        fig.add_vline(
            x=bt.split_date,
            line_dash="dot", line_color="gray", line_width=2,
            annotation_text="Train/Test 분할",
            annotation_position="top left",
        )

    fig.update_layout(
        title="누적 수익률: 최적 포트폴리오 vs 벤치마크",
        xaxis_title="날짜", yaxis_title="$1 투자 시 성장",
        height=480, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 3. Benchmark comparison bar chart
# ---------------------------------------------------------------------------
def benchmark_bar_chart(result: AdvancedOptimizationResult) -> dict:
    """Grouped bars: Sharpe, Sortino, MaxDD across portfolios."""
    bt = result.backtest
    if not bt.benchmark_metrics:
        return {}

    names = ["최적 포트폴리오"] + list(bt.benchmark_metrics.keys())
    sharpes = [bt.sharpe] + [m["sharpe"] for m in bt.benchmark_metrics.values()]
    sortinos = [bt.sortino] + [m["sortino"] for m in bt.benchmark_metrics.values()]
    max_dds = [bt.max_drawdown] + [m["max_drawdown"] for m in bt.benchmark_metrics.values()]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["샤프 비율", "소르티노 비율", "최대 낙폭 (%)"],
    )

    fig.add_trace(go.Bar(x=names, y=sharpes, marker_color="steelblue", name="샤프"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=sortinos, marker_color="coral", name="소르티노"), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=max_dds, marker_color="goldenrod", name="최대 낙폭"), row=1, col=3)

    fig.update_layout(
        title="벤치마크 비교",
        height=400, template="plotly_white",
        showlegend=False,
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 4. Portfolio summary pie with cash
# ---------------------------------------------------------------------------
def portfolio_summary_chart(result: AdvancedOptimizationResult) -> dict:
    """Enhanced pie chart showing asset weights + cash allocation."""
    labels = list(result.weights.keys())
    values = [v * 100 for v in result.weights.values()]

    if result.cash_weight > 1e-6:
        labels.append("현금")
        values.append(result.cash_weight * 100)

    # Filter out zero weights for cleaner display
    filtered = [(l, v) for l, v in zip(labels, values) if abs(v) > 0.01]
    if not filtered:
        return {}
    labels, values = zip(*filtered)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["비중 (파이)", "비중 (막대)"],
    )
    fig.add_trace(
        go.Pie(labels=list(labels), values=list(values),
               textinfo="label+percent", hole=0.35),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=list(labels), y=list(values),
               marker_color="steelblue",
               text=[f"{v:.1f}%" for v in values], textposition="auto"),
        row=1, col=2,
    )
    fig.update_layout(
        title_text="최적 포트폴리오 비중 (현금 포함)",
        height=420, showlegend=False, template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# Bundle all advanced charts
# ---------------------------------------------------------------------------
def generate_advanced_charts(result: AdvancedOptimizationResult) -> dict:
    return {
        "scenario_distribution": scenario_distribution_chart(result),
        "backtest_split": backtest_split_chart(result),
        "benchmark_comparison": benchmark_bar_chart(result),
        "portfolio_summary": portfolio_summary_chart(result),
    }
