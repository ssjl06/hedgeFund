"""
Plotly chart generators for CVaR portfolio optimization results.
Returns JSON strings ready for plotly.js on the frontend.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import OptimizationResult


def _to_json(fig: go.Figure) -> str:
    return json.loads(fig.to_json())


# ---------------------------------------------------------------------------
# 1. Portfolio weights (pie + bar)
# ---------------------------------------------------------------------------
def weights_chart(result: OptimizationResult) -> dict:
    tickers = list(result.weights.keys())
    values = [v * 100 for v in result.weights.values()]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["비중 (파이)", "비중 (막대)"],
    )
    fig.add_trace(
        go.Pie(labels=tickers, values=values, textinfo="label+percent",
               hole=0.35),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=tickers, y=values,
               marker_color="steelblue",
               text=[f"{v:.1f}%" for v in values], textposition="auto"),
        row=1, col=2,
    )
    fig.update_layout(
        title_text="최적 포트폴리오 비중",
        height=420,
        showlegend=False,
        template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 2. Cumulative returns  (portfolio vs benchmarks)
# ---------------------------------------------------------------------------
def cumulative_returns_chart(result: OptimizationResult) -> dict:
    fig = go.Figure()
    idx = result.cumulative_portfolio.index

    fig.add_trace(go.Scatter(
        x=idx, y=result.cumulative_portfolio.values,
        mode="lines", name="최적 포트폴리오",
        line=dict(color="crimson", width=2.5),
    ))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#17becf", "#d62728",
    ]
    for i, col in enumerate(result.cumulative_benchmark.columns):
        dash = "dash" if col == "EqualWeight" else "dot"
        width = 2 if col == "EqualWeight" else 1.2
        label = "동일비중" if col == "EqualWeight" else col
        fig.add_trace(go.Scatter(
            x=idx, y=result.cumulative_benchmark[col].values,
            mode="lines", name=label,
            line=dict(color=colors[i % len(colors)], width=width, dash=dash),
        ))

    fig.update_layout(
        title="누적 수익률: 최적 포트폴리오 vs 벤치마크",
        xaxis_title="날짜", yaxis_title="$1 투자 시 성장",
        height=480, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 3. Efficient frontier
# ---------------------------------------------------------------------------
def efficient_frontier_chart(result: OptimizationResult) -> dict:
    if not result.efficient_frontier:
        return {}

    ef = result.efficient_frontier
    cvars = [p["cvar"] for p in ef]
    rets = [p["return"] for p in ef]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cvars, y=rets,
        mode="lines+markers",
        marker=dict(size=6, color="steelblue"),
        line=dict(color="steelblue", width=2),
        name="효율적 프론티어",
    ))
    # mark the chosen portfolio
    fig.add_trace(go.Scatter(
        x=[result.cvar], y=[result.expected_return],
        mode="markers",
        marker=dict(size=14, color="crimson", symbol="star"),
        name="선택된 포트폴리오",
    ))
    fig.update_layout(
        title="Mean-CVaR 효율적 프론티어",
        xaxis_title="CVaR (%)", yaxis_title="기대 연간 수익률 (%)",
        height=460, template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 4. Return distribution histogram
# ---------------------------------------------------------------------------
def return_distribution_chart(result: OptimizationResult) -> dict:
    port_daily = (result.returns_df * np.array(
        list(result.weights.values()))).sum(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_daily.values * 100,
        nbinsx=80,
        marker_color="steelblue",
        opacity=0.75,
        name="일별 수익률",
    ))

    beta = 0.95  # default
    var_val = np.percentile(port_daily.values, (1 - beta) * 100)
    cvar_val = port_daily[port_daily <= var_val].mean()

    fig.add_vline(x=var_val * 100, line_dash="dash", line_color="orange",
                  annotation_text=f"VaR {beta:.0%}: {var_val*100:.2f}%")
    fig.add_vline(x=cvar_val * 100, line_dash="dash", line_color="red",
                  annotation_text=f"CVaR {beta:.0%}: {cvar_val*100:.2f}%")

    fig.update_layout(
        title="포트폴리오 일별 수익률 분포 (VaR & CVaR)",
        xaxis_title="일별 수익률 (%)", yaxis_title="빈도",
        height=420, template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------
def correlation_chart(result: OptimizationResult) -> dict:
    corr = result.returns_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="종목 간 수익률 상관관계 행렬",
        height=450, template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 6. Drawdown chart
# ---------------------------------------------------------------------------
def drawdown_chart(result: OptimizationResult) -> dict:
    cum = result.cumulative_portfolio
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(220,53,69,0.25)",
        line=dict(color="crimson", width=1.5),
        name="드로우다운",
    ))
    fig.update_layout(
        title="포트폴리오 드로우다운 (최고점 대비 하락폭)",
        xaxis_title="날짜", yaxis_title="드로우다운 (%)",
        height=380, template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# Bundle all charts
# ---------------------------------------------------------------------------
def generate_all_charts(result: OptimizationResult) -> dict:
    return {
        "weights": weights_chart(result),
        "cumulative": cumulative_returns_chart(result),
        "frontier": efficient_frontier_chart(result),
        "distribution": return_distribution_chart(result),
        "correlation": correlation_chart(result),
        "drawdown": drawdown_chart(result),
    }
