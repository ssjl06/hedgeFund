"""
Frontier chart generators for Feature 2: Efficient Frontier Analysis.

Frontier overlay with custom portfolio, weights heatmap,
and lambda profile chart.
"""

import json
import numpy as np
import plotly.graph_objects as go

from frontier import FrontierResult


def _to_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


# ---------------------------------------------------------------------------
# 1. Frontier curve with custom portfolio overlay
# ---------------------------------------------------------------------------
def frontier_overlay_chart(result: FrontierResult) -> dict:
    """
    Frontier curve coloured by lambda + tangent portfolio marker
    + custom portfolio star overlay.
    """
    pts = result.points
    if not pts:
        return {}

    cvars = [p.cvar for p in pts]
    rets = [p.expected_return for p in pts]
    lambdas = [p.lambda_val for p in pts]

    fig = go.Figure()

    # Frontier curve with colour gradient by lambda
    fig.add_trace(go.Scatter(
        x=cvars, y=rets,
        mode="lines+markers",
        marker=dict(
            size=7,
            color=lambdas,
            colorscale="Viridis",
            colorbar=dict(title="λ (원본)", len=0.6),
            showscale=True,
        ),
        line=dict(color="rgba(100,100,100,0.4)", width=1),
        name="효율적 프론티어",
        text=[f"λ={l:.4f}" for l in lambdas],
        hovertemplate="CVaR: %{x:.4f}%<br>수익률: %{y:.2f}%<br>%{text}<extra></extra>",
    ))

    # Tangent portfolio (best Sharpe)
    tang = pts[result.tangent_idx]
    fig.add_trace(go.Scatter(
        x=[tang.cvar], y=[tang.expected_return],
        mode="markers",
        marker=dict(size=16, color="crimson", symbol="diamond"),
        name=f"접선 포트폴리오 (λ={tang.lambda_val:.4f})",
    ))

    # Custom portfolio overlay
    if result.custom_eval:
        ce = result.custom_eval
        fig.add_trace(go.Scatter(
            x=[ce.cvar], y=[ce.expected_return],
            mode="markers",
            marker=dict(size=18, color="gold", symbol="star",
                        line=dict(color="black", width=1.5)),
            name=f"비교 포트폴리오 (Sharpe={ce.sharpe:.3f})",
        ))

    fig.update_layout(
        title="Mean-CVaR 효율적 프론티어 (로그 스케일 λ 스윕)",
        xaxis_title="CVaR (%)", yaxis_title="기대 연간 수익률 (%)",
        height=500, template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 2. Weights heatmap across the frontier
# ---------------------------------------------------------------------------
def frontier_weights_heatmap(result: FrontierResult) -> dict:
    """How weights shift across the frontier (x: frontier index, y: asset, colour: weight)."""
    pts = result.points
    if not pts:
        return {}

    tickers = result.tickers
    n_pts = len(pts)
    z = np.zeros((len(tickers), n_pts))
    for j, pt in enumerate(pts):
        for i, t in enumerate(tickers):
            z[i, j] = pt.weights.get(t, 0.0) * 100

    x_labels = [f"{p.lambda_val:.3g}" for p in pts]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=tickers,
        colorscale="Blues",
        colorbar=dict(title="비중 (%)"),
        text=np.round(z, 1),
        texttemplate="%{text}",
        hovertemplate="λ=%{x}<br>종목: %{y}<br>비중: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="프론티어에 따른 종목 비중 변화",
        xaxis_title="λ (위험 회피도)",
        yaxis_title="종목",
        height=max(350, 40 * len(tickers) + 100),
        template="plotly_white",
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# 3. Lambda profile chart: return / CVaR / volatility vs lambda
# ---------------------------------------------------------------------------
def frontier_lambda_chart(result: FrontierResult) -> dict:
    """Return, CVaR, volatility vs lambda on a log-scale x-axis."""
    pts = result.points
    if not pts:
        return {}

    lambdas = [p.lambda_val for p in pts]
    rets = [p.expected_return for p in pts]
    cvars = [p.cvar for p in pts]
    vols = [p.volatility for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lambdas, y=rets,
        mode="lines+markers",
        marker=dict(size=5), line=dict(width=2, color="steelblue"),
        name="기대 수익률 (%)",
    ))
    fig.add_trace(go.Scatter(
        x=lambdas, y=cvars,
        mode="lines+markers",
        marker=dict(size=5), line=dict(width=2, color="crimson"),
        name="CVaR (%)",
    ))
    fig.add_trace(go.Scatter(
        x=lambdas, y=vols,
        mode="lines+markers",
        marker=dict(size=5), line=dict(width=2, color="goldenrod"),
        name="변동성 (%)",
    ))

    fig.update_layout(
        title="위험 회피도(λ)에 따른 수익/위험 프로필",
        xaxis_title="λ (로그 스케일)",
        xaxis_type="log",
        yaxis_title="값 (%)",
        height=440, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# Bundle all frontier charts
# ---------------------------------------------------------------------------
def generate_frontier_charts(result: FrontierResult) -> dict:
    return {
        "frontier_overlay": frontier_overlay_chart(result),
        "weights_heatmap": frontier_weights_heatmap(result),
        "lambda_profile": frontier_lambda_chart(result),
    }
