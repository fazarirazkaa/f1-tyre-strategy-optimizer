"""Stage 5: Interactive Formula 1 Tyre Degradation & Pit Strategy Optimizer Dashboard.

Built with Streamlit and Plotly following the AQX Sports Analytics Data Bowl 4.0 framework:
1. Practical Application: Real-time 1-stop vs. 2-stop strategy simulator with circuit-specific
   empirical pit loss, FIA 2-compound rule validation, and sensitivity tipping-point maps.
2. Analytical Insight: Fuel-corrected degradation curves and 5-Fold Leave-One-Race-Out CV.
3. Data Presentation: One chart = one message, conclusion-first action titles, authentic F1
   tyre compound colors (Red = SOFT, Yellow = MEDIUM, White = HARD), and plain-language
   takeaways for non-technical audiences.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is importable when running `streamlit run app/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load import DEFAULT_RACES_2024, DRY_COMPOUNDS, PROCESSED_DIR, RANDOM_SEED
from src.simulate import (
    CLIFF_ONSET_SCALED_LAPS,
    MIN_STINT_LAPS,
    CircuitStrategySimulator,
    build_all_circuit_simulators,
    validate_dry_strategy,
)

np.random.seed(RANDOM_SEED)

st.set_page_config(
    page_title="F1 Tyre Degradation & Pit Strategy Optimizer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentic Formula 1 Pirelli dry tyre color palette on dark broadcast canvas
F1_TYRE_COLORS: Dict[str, str] = {
    "SOFT": "#FF2E4C",    # Pirelli Red
    "MEDIUM": "#FFD700",  # Pirelli Yellow
    "HARD": "#FFFFFF",    # Pirelli White
}

PLOT_BG_COLOR = "#141824"
PAPER_BG_COLOR = "#0E1117"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"


def apply_broadcast_layout(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 460,
) -> go.Figure:
    """Apply consistent high-contrast F1 strategy broadcast styling to a Plotly figure."""
    full_title = (
        f"<b>{title}</b><br><span style='font-size:13px;color:#A0AEC0'>{subtitle}</span>"
        if subtitle
        else f"<b>{title}</b>"
    )
    fig.update_layout(
        title=dict(text=full_title, x=0.01, xanchor="left", font=dict(size=17, color="#F8F9FA")),
        template="plotly_dark",
        paper_bgcolor=PAPER_BG_COLOR,
        plot_bgcolor=PLOT_BG_COLOR,
        height=height,
        margin=dict(l=55, r=30, t=78, b=50),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12.5, color="#E2E8F0"),
        legend=dict(
            bgcolor="rgba(20, 24, 36, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.15)",
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


@st.cache_resource(show_spinner="Initializing F1 Telemetry & Strategy Engine...")
def load_dashboard_bundle() -> Dict[str, object]:
    """Load processed datasets and precomputed circuit strategy simulators.

    Automatically executes Stages 2–4 from scratch if processed files are missing,
    guaranteeing full reproducibility on a fresh clone.
    """
    required_files = [
        "clean_laps_2024.csv",
        "race_audit_2024.csv",
        "pit_loss_summary_2024.csv",
        "pit_events_2024.csv",
        "driver_stints_2024.csv",
        "fuel_calibration_2024.csv",
        "baseline_circuit_compound_slopes_2024.csv",
        "compound_initial_offsets_2024.csv",
        "loro_cv_fold_metrics_2024.csv",
        "loro_oof_predictions_2024.csv",
        "error_analysis_by_regime_2024.csv",
        "error_analysis_worst_stints_2024.csv",
        "strategy_optimal_summary_2024.csv",
        "strategy_top_candidates_2024.csv",
        "strategy_sensitivity_grid_2024.csv",
    ]
    if not all((PROCESSED_DIR / f).exists() for f in required_files):
        from src.load import run_stage2_pipeline
        from src.model import run_stage3_pipeline
        from src.simulate import run_stage4_pipeline

        run_stage2_pipeline(save_processed=True)
        run_stage3_pipeline(save_outputs=True)
        run_stage4_pipeline(save_outputs=True)

    clean_df = pd.read_csv(PROCESSED_DIR / "clean_laps_2024.csv")
    audit_df = pd.read_csv(PROCESSED_DIR / "race_audit_2024.csv")
    pit_summary_df = pd.read_csv(PROCESSED_DIR / "pit_loss_summary_2024.csv")
    pit_events_df = pd.read_csv(PROCESSED_DIR / "pit_events_2024.csv")
    stints_df = pd.read_csv(PROCESSED_DIR / "driver_stints_2024.csv")
    fuel_calib_df = pd.read_csv(PROCESSED_DIR / "fuel_calibration_2024.csv")
    baseline_table = pd.read_csv(PROCESSED_DIR / "baseline_circuit_compound_slopes_2024.csv")
    compound_offsets_df = pd.read_csv(PROCESSED_DIR / "compound_initial_offsets_2024.csv")
    loro_metrics_df = pd.read_csv(PROCESSED_DIR / "loro_cv_fold_metrics_2024.csv")
    oof_df = pd.read_csv(PROCESSED_DIR / "loro_oof_predictions_2024.csv")
    error_regimes_df = pd.read_csv(PROCESSED_DIR / "error_analysis_by_regime_2024.csv")
    worst_stints_df = pd.read_csv(PROCESSED_DIR / "error_analysis_worst_stints_2024.csv")
    optimal_summary_df = pd.read_csv(PROCESSED_DIR / "strategy_optimal_summary_2024.csv")
    top_strategies_df = pd.read_csv(PROCESSED_DIR / "strategy_top_candidates_2024.csv")
    sensitivity_df = pd.read_csv(PROCESSED_DIR / "strategy_sensitivity_grid_2024.csv")

    simulators = build_all_circuit_simulators(clean_df=clean_df)

    return {
        "clean_laps": clean_df,
        "audit": audit_df,
        "pit_summary": pit_summary_df,
        "pit_events": pit_events_df,
        "stints": stints_df,
        "fuel_calib": fuel_calib_df,
        "baseline_table": baseline_table,
        "compound_offsets": compound_offsets_df,
        "loro_metrics": loro_metrics_df,
        "oof": oof_df,
        "error_regimes": error_regimes_df,
        "worst_stints": worst_stints_df,
        "optimal_summary": optimal_summary_df,
        "top_strategies": top_strategies_df,
        "sensitivity": sensitivity_df,
        "simulators": simulators,
    }


def render_sidebar(bundle: Dict[str, object]) -> str:
    """Render sidebar circuit selector and empirical circuit telemetry card."""
    st.sidebar.markdown("## 🏎️ Race Control & Circuit")
    circuits = list(DEFAULT_RACES_2024.keys())
    circuit_labels = {k: f"{k} ({v})" for k, v in DEFAULT_RACES_2024.items()}

    selected_circuit = st.sidebar.selectbox(
        "Select 2024 Grand Prix Circuit",
        options=circuits,
        index=1,  # Default to Spain (classic strategic benchmark)
        format_func=lambda k: circuit_labels[k],
    )

    sim: CircuitStrategySimulator = bundle["simulators"][selected_circuit]
    opt_df: pd.DataFrame = bundle["optimal_summary"]
    opt_row = opt_df[opt_df["Circuit"] == selected_circuit].iloc[0]
    pit_df: pd.DataFrame = bundle["pit_summary"]
    pit_row = pit_df[pit_df["Circuit"] == selected_circuit].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📍 {sim.event_name}")
    st.sidebar.markdown(
        f"""
        - **Race Distance:** `{sim.total_laps} Laps`
        - **Empirical Pit Loss:** **`{sim.empirical_pit_loss:.2f} s`** *(IQR `{sim.pit_loss_iqr:.2f} s`, `n={int(pit_row['ValidPitStops'])}` green stops)*
        - **Fuel Weight Penalty:** `{sim.fuel_penalty_s_per_lap:.4f} s/lap` *(`{sim.fuel_burn_kg_per_lap:.2f} kg/lap`)*
        - **Track / Air Temp:** `{sim.track_temp:.1f} °C` / `{sim.air_temp:.1f} °C`
        - **Optimal Strategy:** **`{opt_row['RecommendedStrategy']}`** (`{opt_row['OptimalCompounds']}`, Pit Lap `{opt_row['OptimalPitLaps']}`)
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Pirelli Dry Tyre Legend:**
        - 🔴 **SOFT** (`#FF2E4C`): Highest grip, fastest wear
        - 🟡 **MEDIUM** (`#FFD700`): Balanced pace & durability
        - ⚪ **HARD** (`#FFFFFF`): Lowest wear, slowest initial warmup
        """
    )
    return selected_circuit


def render_tab_summary(bundle: Dict[str, object], selected_circuit: str) -> None:
    """Render Tab 1: Executive Summary with plain-language core takeaway and cross-circuit comparison."""
    opt_df: pd.DataFrame = bundle["optimal_summary"].copy()
    pit_df: pd.DataFrame = bundle["pit_summary"]
    sim: CircuitStrategySimulator = bundle["simulators"][selected_circuit]
    opt_row = opt_df[opt_df["Circuit"] == selected_circuit].iloc[0]

    # Plain-language key finding banner for non-technical audiences
    st.info(
        "### 💡 Primary Finding in Plain Language\n"
        "**Winning pit strategy is a mathematical tug-of-war between Pit Lane Time Loss (20.6–24.2 seconds lost per stop) "
        "and Tyre Wear Rate (0.04–0.12 seconds lost per lap on aging rubber).**\n\n"
        "- **When to 2-Stop (Hungary & Spain):** At circuits with a short pit lane (**Hungary: 20.60 s pit loss**) or high thermal "
        "abrasion (**Spain: 0.091 s/lap Soft wear**), fresh tyres easily recover the cost of an extra stop—making a **2-stop strategy "
        "1.09 s to 6.70 s faster** over the race distance.\n"
        "- **When to 1-Stop (Abu Dhabi, Netherlands & Japan):** When pit stops cost **23.0–24.2 s** and the Hard compound degrades "
        "gently (**0.041–0.051 s/lap**), a single pit stop around **43%–46% race distance (`MEDIUM → HARD`) saves 2.04 s to 6.52 s**."
    )

    gap_val = float(opt_row["TimeGap_2Stop_vs_1Stop_s"])
    if gap_val < 0:
        winner_badge = f"2-Stop Faster by {abs(gap_val):.2f} s"
    else:
        winner_badge = f"1-Stop Faster by {abs(gap_val):.2f} s"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label=f"{selected_circuit} Recommended Strategy",
        value=str(opt_row["RecommendedStrategy"]),
        delta=winner_badge,
    )
    col2.metric(
        label="Optimal Compound Sequence",
        value=str(opt_row["OptimalCompounds"]),
        delta=f"Stints: {opt_row['OptimalStintLengths']} laps",
        delta_color="off",
    )
    col3.metric(
        label="Optimal Pit Stop Window",
        value=f"Lap {opt_row['OptimalPitLaps']}",
        delta=f"Total Race Time: {opt_row['OptimalTotalTime']}",
        delta_color="off",
    )
    col4.metric(
        label="Empirical Green-Flag Pit Loss",
        value=f"{sim.empirical_pit_loss:.2f} s",
        delta=f"IQR: {sim.pit_loss_iqr:.2f} s (from telemetry)",
        delta_color="off",
    )

    st.markdown("---")

    # Chart 1: Cross-circuit 1-stop vs 2-stop time advantage (One chart, one clear message)
    opt_df["TwoStopGainSeconds"] = -opt_df["TimeGap_2Stop_vs_1Stop_s"]
    opt_df = opt_df.sort_values("TwoStopGainSeconds", ascending=True)
    bar_colors = [
        "#00E676" if val > 0 else "#38BDF8" for val in opt_df["TwoStopGainSeconds"]
    ]

    fig_summary = go.Figure(
        go.Bar(
            x=opt_df["TwoStopGainSeconds"],
            y=opt_df["Circuit"]
            + " (Pit Loss: "
            + opt_df["EmpiricalPitLossSeconds"].map(lambda v: f"{v:.1f}s")
            + ")",
            orientation="h",
            marker_color=bar_colors,
            text=[
                f"2-Stop wins by +{v:.2f}s ({c})"
                if v > 0
                else f"1-Stop wins by +{abs(v):.2f}s ({c})"
                for v, c in zip(opt_df["TwoStopGainSeconds"], opt_df["OptimalCompounds"])
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Optimal Strategy: %{customdata[0]} (%{customdata[1]})<br>"
                "Optimal Pit Lap(s): %{customdata[2]}<br>"
                "2-Stop vs 1-Stop Advantage: %{x:+.2f} s<extra></extra>"
            ),
            customdata=opt_df[
                ["RecommendedStrategy", "OptimalCompounds", "OptimalPitLaps"]
            ].to_numpy(),
        )
    )
    fig_summary.add_vline(x=0, line_width=2, line_dash="dash", line_color="#CBD5E1")
    fig_summary.update_xaxes(
        title_text="Net Race Time Advantage of 2-Stop vs. 1-Stop (seconds)  [ ← 1-Stop Faster | 2-Stop Faster → ]",
        range=[-10.5, 10.5],
    )
    fig_summary.update_yaxes(title_text="2024 Grand Prix Circuit & Empirical Pit Loss")
    apply_broadcast_layout(
        fig_summary,
        title="2-Stop Beats 1-Stop by 6.7s at Short-Pit Hungary, While 1-Stop Wins by 6.5s at Low-Wear Abu Dhabi",
        subtitle="Comparison of optimal 1-stop vs. 2-stop total race time across all five 2024 dry Grand Prix circuits",
        height=420,
    )
    st.plotly_chart(fig_summary, width="stretch")

    # Summary comparison table across all 5 circuits
    st.markdown("#### 📊 Cross-Circuit Strategic Decision Table (2024 Dry Races)")
    display_cols = [
        "Circuit",
        "EventName",
        "TotalRaceLaps",
        "EmpiricalPitLossSeconds",
        "RecommendedStrategy",
        "OptimalCompounds",
        "OptimalPitLaps",
        "Best1StopCompounds",
        "Best1StopPitLap",
        "Best1StopTime",
        "Best2StopCompounds",
        "Best2StopPitLaps",
        "Best2StopTime",
    ]
    st.dataframe(
        bundle["optimal_summary"][display_cols].merge(
            pit_df[["Circuit", "IQRPitLossSeconds", "ValidPitStops"]],
            on="Circuit",
            how="left",
        ),
        width="stretch",
        hide_index=True,
    )


def render_tab_degradation(bundle: Dict[str, object], selected_circuit: str) -> None:
    """Render Tab 2: Tyre Degradation Curves per Compound (F1 colors: Red/Yellow/White)."""
    clean_df: pd.DataFrame = bundle["clean_laps"]
    sim: CircuitStrategySimulator = bundle["simulators"][selected_circuit]
    baseline_table: pd.DataFrame = bundle["baseline_table"]

    c_df = clean_df[clean_df["Circuit"] == selected_circuit].copy()
    c_slopes = baseline_table[baseline_table["Circuit"] == selected_circuit].set_index("Compound")

    s_slope = sim.circuit_slopes["SOFT"]
    m_slope = sim.circuit_slopes["MEDIUM"]
    h_slope = sim.circuit_slopes["HARD"]

    # Empirical median observations by TyreLife + continuous model curve per compound
    binned = (
        c_df.groupby(["Compound", "TyreLife"], as_index=False)
        .agg(
            MedianWear=("StintDegradationDelta", "median"),
            Count=("LapNumber", "count"),
        )
    )
    binned = binned[binned["Count"] >= 2]

    fig_deg = go.Figure()
    max_plot_tl = min(sim.total_laps, 42)
    tl_grid = np.arange(1, max_plot_tl + 1, dtype=int)
    mid_race_lap_idx = min(sim.total_laps - 1, sim.total_laps // 2)

    for comp in ("SOFT", "MEDIUM", "HARD"):
        # Empirical data points
        sub_b = binned[binned["Compound"] == comp]
        if not sub_b.empty:
            fig_deg.add_trace(
                go.Scatter(
                    x=sub_b["TyreLife"],
                    y=sub_b["MedianWear"] + sim.fresh_offsets[comp],
                    mode="markers",
                    name=f"{comp} (Observed Telemetry Median)",
                    legendgroup=comp,
                    marker=dict(
                        color=F1_TYRE_COLORS[comp],
                        size=8,
                        opacity=0.8,
                        line=dict(color="#0E1117", width=1),
                    ),
                    hovertemplate=(
                        f"<b>{comp} (Empirical)</b><br>"
                        "Tyre Age: %{x} laps<br>"
                        "Fuel-Adjusted Pace Delta: %{y:+.2f} s<extra></extra>"
                    ),
                )
            )

        # Model curve (fresh compound offset + precomputed wear lookup at mid-race fuel load)
        max_comp_tl = 26 if comp == "SOFT" else (35 if comp == "MEDIUM" else max_plot_tl)
        tl_comp = tl_grid[tl_grid <= max_comp_tl]
        wear_curve = (
            sim.fresh_offsets[comp]
            + sim.wear_lookup[comp][mid_race_lap_idx, tl_comp - 1]
        )
        fig_deg.add_trace(
            go.Scatter(
                x=tl_comp,
                y=wear_curve,
                mode="lines",
                name=f"{comp} Model Curve ({sim.circuit_slopes[comp]:.3f} s/lap)",
                legendgroup=comp,
                line=dict(color=F1_TYRE_COLORS[comp], width=3.5),
                hovertemplate=(
                    f"<b>{comp} (Model Curve)</b><br>"
                    "Tyre Age: %{x} laps<br>"
                    "Predicted Pace Delta vs Fresh Medium: %{y:+.2f} s<extra></extra>"
                ),
            )
        )

    fig_deg.update_xaxes(title_text="Tyre Age (Laps Completed on Set)", range=[0.5, max_plot_tl + 1])
    fig_deg.update_yaxes(
        title_text="Fuel-Adjusted Lap Time Delta vs. Fresh Medium (seconds, lower = faster)"
    )
    apply_broadcast_layout(
        fig_deg,
        title=f"At {selected_circuit}, Soft Degrades at {s_slope:.3f} s/lap vs. {m_slope:.3f} s/lap for Medium and {h_slope:.3f} s/lap for Hard",
        subtitle=(
            "Fuel-corrected & driver-normalized pace delta by tyre age (Red = SOFT, Yellow = MEDIUM, White = HARD). "
            "Soft starts ~0.23s faster when fresh but crosses Medium around Lap 10–13."
        ),
        height=500,
    )
    st.plotly_chart(fig_deg, width="stretch")

    # Second chart: Why Fuel Correction is Essential (Raw vs Fuel-Adjusted)
    col_left, col_right = st.columns([3, 2])
    with col_left:
        lap_med = (
            c_df.groupby(["LapNumber", "Compound"], as_index=False)
            .agg(
                RawLap=("LapTimeSeconds", "median"),
                FuelAdjLap=("CircuitNormalizedLapTime", "median"),
                MeanTyreLife=("TyreLife", "mean"),
            )
        )
        fig_fuel = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Raw Lap Time vs. Race Lap (Fuel Burn Masks Wear)",
                "Fuel-Adjusted Lap Time vs. Tyre Age (True Wear)",
            ),
        )
        for comp in ("SOFT", "MEDIUM", "HARD"):
            sub = lap_med[lap_med["Compound"] == comp]
            if sub.empty:
                continue
            fig_fuel.add_trace(
                go.Scatter(
                    x=sub["LapNumber"],
                    y=sub["RawLap"],
                    mode="markers",
                    name=comp,
                    legendgroup=comp,
                    marker=dict(color=F1_TYRE_COLORS[comp], size=6.5),
                ),
                row=1,
                col=1,
            )
            fig_fuel.add_trace(
                go.Scatter(
                    x=sub["MeanTyreLife"],
                    y=sub["FuelAdjLap"],
                    mode="markers",
                    name=comp,
                    legendgroup=comp,
                    showlegend=False,
                    marker=dict(color=F1_TYRE_COLORS[comp], size=6.5, symbol="diamond"),
                ),
                row=1,
                col=2,
            )
        fig_fuel.update_xaxes(title_text="Race Lap Number", row=1, col=1)
        fig_fuel.update_yaxes(title_text="Median Raw Lap Time (s)", row=1, col=1)
        fig_fuel.update_xaxes(title_text="Mean Tyre Life (Laps)", row=1, col=2)
        fig_fuel.update_yaxes(title_text="Fuel-Adjusted Lap Time (s)", row=1, col=2)
        apply_broadcast_layout(
            fig_fuel,
            title=f"Removing the {sim.fuel_penalty_s_per_lap:.3f} s/lap Fuel Burn Effect at {selected_circuit} Unmasks Positive Tyre Wear",
            subtitle="Cars burn ~105 kg of fuel over a race (~3.2s lighter at the finish), making raw lap times appear flat or decreasing",
            height=420,
        )
        st.plotly_chart(fig_fuel, width="stretch")

    with col_right:
        st.markdown(f"#### 🛞 {selected_circuit} Compound Parameters")
        comp_summary_rows = []
        for comp in ("SOFT", "MEDIUM", "HARD"):
            row_b = c_slopes.loc[comp]
            comp_summary_rows.append(
                {
                    "Compound": comp,
                    "Stints": int(row_b["Stints"]),
                    "Clean Laps": int(row_b["CleanLaps"]),
                    "Fresh Offset vs Med (s)": f"{sim.fresh_offsets[comp]:+.3f} s",
                    "Wear Rate (s/lap)": f"+{sim.circuit_slopes[comp]:.4f} s/lap",
                    "Cliff Onset (Laps)": int(
                        round(CLIFF_ONSET_SCALED_LAPS[comp] / sim.lap_scale)
                    ),
                }
            )
        st.dataframe(pd.DataFrame(comp_summary_rows), width="stretch", hide_index=True)
        st.markdown(
            f"""
            **Key Analytical Insights for {selected_circuit}:**
            - **Fresh Grip vs. Wear Trade-off:** `SOFT` starts `{abs(sim.fresh_offsets['SOFT']):.2f} s/lap` faster than `MEDIUM`, while `HARD` starts `{sim.fresh_offsets['HARD']:+.2f} s/lap` slower than `MEDIUM`.
            - **Crossover Window:** `SOFT` loses its per-lap pace advantage over `MEDIUM` after **`~{int(abs(sim.fresh_offsets['SOFT']) / max(s_slope - m_slope, 0.01))} laps`**, and `MEDIUM` crosses `HARD` after **`~{int(sim.fresh_offsets['HARD'] / max(m_slope - h_slope, 0.008))} laps`**.
            """
        )


def render_tab_timeline(bundle: Dict[str, object], selected_circuit: str) -> None:
    """Render Tab 3: Actual 2024 Driver Strategy Timeline (Horizontal Bar Chart per Stint)."""
    stints_df: pd.DataFrame = bundle["stints"]
    c_stints = stints_df[stints_df["Circuit"] == selected_circuit].copy()

    # Order drivers from P1 (top) to P20 (bottom)
    driver_order_df = (
        c_stints[["Driver", "FinalPosition", "Team"]]
        .drop_duplicates(subset=["Driver"])
        .sort_values("FinalPosition", ascending=False)
    )
    driver_labels = [
        f"P{int(row['FinalPosition']):02d} {row['Driver']} ({row['Team']})"
        for _, row in driver_order_df.iterrows()
    ]
    driver_to_label = dict(zip(driver_order_df["Driver"], driver_labels))
    c_stints["DriverLabel"] = c_stints["Driver"].map(driver_to_label)

    # Identify winner's strategy for conclusion title
    p1_driver = driver_order_df.iloc[-1]["Driver"]
    p1_stints = c_stints[c_stints["Driver"] == p1_driver].sort_values("Stint")
    p1_seq = " → ".join(p1_stints["Compound"].tolist())
    p1_stops = max(0, len(p1_stints) - 1)

    fig_timeline = go.Figure()
    shown_compounds = set()

    for _, row in c_stints.sort_values(["FinalPosition", "Stint"], ascending=[False, True]).iterrows():
        comp = str(row["Compound"])
        stint_len = int(row["EndLap"]) - int(row["StartLap"]) + 1
        start_lap = int(row["StartLap"]) - 1
        show_in_legend = comp not in shown_compounds
        shown_compounds.add(comp)

        fresh_tag = "Fresh Set" if bool(row["FreshTyre"]) else f"Used Set (Start Age: {int(row['InitialTyreLife'])}L)"
        med_lt = (
            f"{float(row['MedianLapTimeSeconds']):.3f} s"
            if pd.notna(row["MedianLapTimeSeconds"])
            else "N/A"
        )

        fig_timeline.add_trace(
            go.Bar(
                y=[row["DriverLabel"]],
                x=[stint_len],
                base=[start_lap],
                orientation="h",
                name=comp,
                legendgroup=comp,
                showlegend=show_in_legend,
                marker=dict(
                    color=F1_TYRE_COLORS.get(comp, "#888888"),
                    line=dict(color="#0E1117", width=1.8),
                ),
                text=f"{comp[0]} ({stint_len}L)" if stint_len >= 5 else comp[0],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    color="#0E1117" if comp in ("MEDIUM", "HARD") else "#FFFFFF",
                    size=11,
                ),
                hovertemplate=(
                    f"<b>{row['Driver']} ({row['Team']}) — Stint {int(row['Stint'])}</b><br>"
                    f"Compound: <b>{comp}</b> ({fresh_tag})<br>"
                    f"Laps: Lap {int(row['StartLap'])} to Lap {int(row['EndLap'])} ({stint_len} laps)<br>"
                    f"Max Tyre Life Reached: {int(row['MaxTyreLife'])} laps<br>"
                    f"Median Clean Lap Time: {med_lt}<extra></extra>"
                ),
            )
        )

    fig_timeline.update_layout(barmode="stack")
    fig_timeline.update_xaxes(
        title_text="Race Lap Number (Start to Chequered Flag)",
        range=[0, int(c_stints["EndLap"].max()) + 1],
    )
    fig_timeline.update_yaxes(
        title_text="Finishing Order (P1 at Top)",
        categoryorder="array",
        categoryarray=driver_labels,
    )
    apply_broadcast_layout(
        fig_timeline,
        title=f"2024 {selected_circuit} GP Strategy Execution: Winner {p1_driver} Executed a {p1_stops}-Stop ({p1_seq}) Strategy",
        subtitle="Horizontal bars show each driver's actual tyre compound stints and pit stop laps (Red = SOFT, Yellow = MEDIUM, White = HARD)",
        height=650,
    )
    st.plotly_chart(fig_timeline, width="stretch")


def render_tab_simulator(bundle: Dict[str, object], selected_circuit: str) -> None:
    """Render Tab 4: Interactive Pit Strategy Simulator & Two-Way Sensitivity Analysis."""
    sim: CircuitStrategySimulator = bundle["simulators"][selected_circuit]
    n_laps = sim.total_laps

    st.markdown(
        f"### 🛠️ Interactive Pit Stop Strategy Simulator — **{sim.event_name} ({n_laps} Laps)**"
    )
    st.caption(
        "Configure your starting tyre compound, subsequent stint compounds, pit stop lap(s), and sensitivity parameters below. "
        "Note: FIA Dry Race Regulations (Article 30.5m) require using at least two different dry compounds."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.8, 1.5])

    with ctrl1:
        num_stops = st.radio(
            "Select Strategy Type",
            options=[1, 2],
            format_func=lambda x: f"{x}-Stop Strategy ({x + 1} Stints)",
            horizontal=True,
        )
        start_comp = st.selectbox(
            "Stint 1 (Starting Compound)",
            options=list(DRY_COMPOUNDS),
            index=1,  # Default MEDIUM
        )
        stint2_comp = st.selectbox(
            "Stint 2 Compound",
            options=list(DRY_COMPOUNDS),
            index=2 if num_stops == 1 else 0,  # Default HARD for 1-stop, SOFT for 2-stop
        )
        if num_stops == 2:
            stint3_comp = st.selectbox(
                "Stint 3 Compound",
                options=list(DRY_COMPOUNDS),
                index=2,  # Default HARD
            )
            chosen_compounds = (start_comp, stint2_comp, stint3_comp)
        else:
            chosen_compounds = (start_comp, stint2_comp)

    with ctrl2:
        if num_stops == 1:
            default_p1 = min(max(MIN_STINT_LAPS, int(round(n_laps * 0.44))), n_laps - MIN_STINT_LAPS)
            p1 = st.slider(
                "Pit Stop 1 Lap",
                min_value=MIN_STINT_LAPS,
                max_value=n_laps - MIN_STINT_LAPS,
                value=default_p1,
                step=1,
            )
            chosen_pits = (p1,)
        else:
            default_p1 = min(max(MIN_STINT_LAPS, int(round(n_laps * 0.34))), n_laps - 2 * MIN_STINT_LAPS)
            p1 = st.slider(
                "Pit Stop 1 Lap",
                min_value=MIN_STINT_LAPS,
                max_value=n_laps - 2 * MIN_STINT_LAPS,
                value=default_p1,
                step=1,
            )
            default_p2 = min(max(p1 + MIN_STINT_LAPS, int(round(n_laps * 0.65))), n_laps - MIN_STINT_LAPS)
            p2 = st.slider(
                "Pit Stop 2 Lap",
                min_value=p1 + MIN_STINT_LAPS,
                max_value=n_laps - MIN_STINT_LAPS,
                value=default_p2,
                step=1,
            )
            chosen_pits = (p1, p2)

    with ctrl3:
        custom_pit_loss = st.slider(
            f"Pit Stop Loss (s) [Empirical Median = {sim.empirical_pit_loss:.2f}s]",
            min_value=15.0,
            max_value=32.0,
            value=float(sim.empirical_pit_loss),
            step=0.25,
        )
        deg_mult = st.slider(
            "Tyre Degradation Rate Multiplier (1.00x = Empirical Baseline)",
            min_value=0.70,
            max_value=1.50,
            value=1.00,
            step=0.05,
        )

    # Validate FIA 2-compound rule
    is_valid, valid_msg = validate_dry_strategy(chosen_compounds, chosen_pits, n_laps, min_stint_laps=2)

    # Search optimal strategies under the current sensitivity sliders
    ranked_all = sim.search_optimal_strategies(
        pit_loss_override=custom_pit_loss,
        degradation_multiplier=deg_mult,
    )
    opt_overall = ranked_all.iloc[0]
    opt_1stop = ranked_all[ranked_all["NumStops"] == 1].iloc[0]
    opt_2stop = ranked_all[ranked_all["NumStops"] == 2].iloc[0]

    if not is_valid:
        st.error(
            f"⚠️ **Rule Violation:** {valid_msg} "
            f"Under FIA Sporting Regulations Article 30.5(m), a dry race strategy must use at least two different tyre compounds. "
            f"Please change one of your stint compounds (e.g., `{opt_overall['CompoundSequence']}`)."
        )
        return

    user_sim = sim.simulate_strategy(
        compounds=chosen_compounds,
        pit_laps=chosen_pits,
        pit_loss_override=custom_pit_loss,
        degradation_multiplier=deg_mult,
        return_lap_trace=True,
    )
    opt1_trace = sim.simulate_strategy(
        compounds=tuple(str(opt_1stop["CompoundSequence"]).split(" → ")),
        pit_laps=(int(opt_1stop["PitLap1"]),),
        pit_loss_override=custom_pit_loss,
        degradation_multiplier=deg_mult,
        return_lap_trace=True,
    )
    opt2_pits = [int(x.strip()) for x in str(opt_2stop["PitLaps"]).split(",")]
    opt2_trace = sim.simulate_strategy(
        compounds=tuple(str(opt_2stop["CompoundSequence"]).split(" → ")),
        pit_laps=tuple(opt2_pits),
        pit_loss_override=custom_pit_loss,
        degradation_multiplier=deg_mult,
        return_lap_trace=True,
    )

    user_time = float(user_sim["TotalRaceSeconds"])
    opt_time = float(opt_overall["TotalRaceSeconds"])
    delta_vs_opt = user_time - opt_time

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Your Selected Strategy Time",
        str(user_sim["FormattedRaceTime"]),
        delta="Optimal!" if delta_vs_opt <= 0.05 else f"+{delta_vs_opt:.2f} s slower than optimal",
        delta_color="normal" if delta_vs_opt <= 0.05 else "inverse",
    )
    k2.metric(
        f"Best 1-Stop ({opt_1stop['CompoundSequence']} | L{opt_1stop['PitLaps']})",
        str(opt_1stop["FormattedRaceTime"]),
        delta=f"{float(opt_1stop['TotalRaceSeconds']) - opt_time:+.2f} s vs overall best",
        delta_color="off",
    )
    k3.metric(
        f"Best 2-Stop ({opt_2stop['CompoundSequence']} | L{opt_2stop['PitLaps']})",
        str(opt_2stop["FormattedRaceTime"]),
        delta=f"{float(opt_2stop['TotalRaceSeconds']) - opt_time:+.2f} s vs overall best",
        delta_color="off",
    )
    two_vs_one = float(opt_1stop["TotalRaceSeconds"]) - float(opt_2stop["TotalRaceSeconds"])
    k4.metric(
        "1-Stop vs. 2-Stop Verdict",
        "2-Stop Wins" if two_vs_one > 0 else "1-Stop Wins",
        delta=f"by {abs(two_vs_one):.2f} seconds",
        delta_color="normal",
    )

    # Chart 4A & 4B side by side: Cumulative Gap Trace + 1-Stop Pit Window Curve
    c_left, c_right = st.columns(2)

    with c_left:
        # Plot cumulative time gap relative to Optimal 1-Stop across all laps
        df_u = user_sim["LapTrace"]
        df_1 = opt1_trace["LapTrace"]
        df_2 = opt2_trace["LapTrace"]

        ref_cum = df_1["CumulativeRaceSeconds"].to_numpy(dtype=float)
        gap_user = ref_cum - df_u["CumulativeRaceSeconds"].to_numpy(dtype=float)
        gap_opt2 = ref_cum - df_2["CumulativeRaceSeconds"].to_numpy(dtype=float)

        fig_trace = go.Figure()
        fig_trace.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#38BDF8",
            annotation_text=f"Benchmark 1-Stop ({opt_1stop['CompoundSequence']} | L{opt_1stop['PitLaps']})",
            annotation_position="top left",
        )
        fig_trace.add_trace(
            go.Scatter(
                x=df_2["LapNumber"],
                y=gap_opt2,
                mode="lines",
                name=f"Optimal 2-Stop ({opt_2stop['CompoundSequence']} | L{opt_2stop['PitLaps']})",
                line=dict(color="#00E676", width=2.8),
            )
        )
        fig_trace.add_trace(
            go.Scatter(
                x=df_u["LapNumber"],
                y=gap_user,
                mode="lines+markers",
                name=f"Your Strategy ({user_sim['CompoundSequence']} | L{user_sim['PitLaps']})",
                line=dict(color="#FF2E4C", width=3.0),
                marker=dict(size=5),
            )
        )
        fig_trace.update_xaxes(title_text="Race Lap Number")
        fig_trace.update_yaxes(
            title_text="Time Advantage vs. Optimal 1-Stop (seconds, + = Ahead / Faster)"
        )
        apply_broadcast_layout(
            fig_trace,
            title="Head-to-Head Cumulative Race Gap vs. Optimal 1-Stop Benchmark",
            subtitle="Sharp drops mark pit stop time losses; upward slopes show pace recovery on fresher tyres",
            height=430,
        )
        st.plotly_chart(fig_trace, width="stretch")

    with c_right:
        # 1-Stop Pit Window Curve across all candidate pit laps
        one_stop_all = ranked_all[ranked_all["NumStops"] == 1].copy()
        top_pairs = (
            one_stop_all.groupby("CompoundSequence")["TotalRaceSeconds"]
            .min()
            .nsmallest(3)
            .index.tolist()
        )
        fig_win = go.Figure()
        pair_palette = ["#38BDF8", "#FFD700", "#FF2E4C"]
        for idx, seq in enumerate(top_pairs):
            sub_seq = one_stop_all[one_stop_all["CompoundSequence"] == seq].sort_values("PitLap1")
            fig_win.add_trace(
                go.Scatter(
                    x=sub_seq["PitLap1"],
                    y=sub_seq["TotalRaceSeconds"] - opt_time,
                    mode="lines",
                    name=f"1-Stop: {seq}",
                    line=dict(color=pair_palette[idx % len(pair_palette)], width=3),
                )
            )
        if num_stops == 1:
            fig_win.add_vline(
                x=chosen_pits[0],
                line_dash="dot",
                line_color="#FFFFFF",
                annotation_text=f"Your Pit Lap ({chosen_pits[0]})",
            )
        fig_win.update_xaxes(title_text="1-Stop Pit Stop Lap")
        fig_win.update_yaxes(title_text="Time Loss vs. Global Optimal Strategy (seconds)")
        apply_broadcast_layout(
            fig_win,
            title=f"1-Stop Pit Window at {selected_circuit} Opens Around Lap {int(opt_1stop['PitLap1']) - 3}–{int(opt_1stop['PitLap1']) + 3}",
            subtitle="Pitting too early triggers late-race tyre cliff; pitting too late wastes fresh-tyre pace",
            height=430,
        )
        st.plotly_chart(fig_win, width="stretch")

    # Chart 4C: Two-Way Sensitivity Heatmap (Pit Loss vs Degradation Multiplier)
    sens_df: pd.DataFrame = bundle["sensitivity"]
    c_sens = sens_df[sens_df["Circuit"] == selected_circuit].copy()
    piv = c_sens.pivot(
        index="DegradationMultiplier",
        columns="PitLossSeconds",
        values="TwoStopAdvantageSeconds",
    )
    text_piv = c_sens.pivot(
        index="DegradationMultiplier",
        columns="PitLossSeconds",
        values="OptimalStrategyType",
    )

    z_vals = piv.to_numpy(dtype=float)
    text_labels = [
        [
            f"{text_piv.iloc[r, c]}<br>({z_vals[r, c]:+.1f}s)"
            for c in range(z_vals.shape[1])
        ]
        for r in range(z_vals.shape[0])
    ]

    fig_heat = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=[f"{col:.1f}s" for col in piv.columns],
            y=[f"{idx:.2f}x" for idx in piv.index],
            text=text_labels,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar=dict(title="2-Stop Gain (s)"),
            hovertemplate=(
                "Pit Stop Loss: %{x}<br>"
                "Degradation Multiplier: %{y}<br>"
                "2-Stop Advantage vs 1-Stop: %{z:+.2f} s<extra></extra>"
            ),
        )
    )
    fig_heat.update_xaxes(title_text="Green-Flag Pit Stop Time Loss (seconds)")
    fig_heat.update_yaxes(title_text="Tyre Degradation Rate Multiplier")
    apply_broadcast_layout(
        fig_heat,
        title=f"Sensitivity Analysis ({selected_circuit}): Green Cells Favor 2-Stop (+s), Red Cells Favor 1-Stop (-s)",
        subtitle="Two-way sensitivity across Pit Stop Loss (+/- 4s) and Tyre Degradation Rate (0.75x to 1.30x)",
        height=440,
    )
    st.plotly_chart(fig_heat, width="stretch")


def render_tab_methodology(bundle: Dict[str, object]) -> None:
    """Render Tab 5: Methodology, Leak-Free LORO-CV Validation & Honest Limitations."""
    loro_df: pd.DataFrame = bundle["loro_metrics"]
    err_df: pd.DataFrame = bundle["error_regimes"]
    audit_df: pd.DataFrame = bundle["audit"]
    fuel_df: pd.DataFrame = bundle["fuel_calib"]

    st.markdown("### 📐 Methodology, Explicit Assumptions & Leak-Free Model Validation")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            """
            #### 1. Data Hygiene & Two-Stage Stint Outlier Filtering (`src/load.py`)
            - **Source:** 2024 FIA Formula One World Championship Race (`'R'`) sessions via **FastF1** (`5,196` clean racing laps retained from `6,037` raw laps across 5 dry races: **Japan, Spain, Hungary, Netherlands, Abu Dhabi**; `86.1%` retention).
            - **Race Selection Audit:** Initial EDA revealed that **2024 Bahrain GP** had `0` Medium stints and **2024 Italian GP (Monza)** had only `1` clean Soft lap. They were replaced with **Japan (Suzuka)** and **Netherlands (Zandvoort)** so every race has all three dry compounds (`SOFT`, `MEDIUM`, `HARD`).
            - **Non-Representative Lap Removal:** Excludes Lap 1 (standing start), Pit In-Laps, Pit Out-Laps, Safety Car / VSC / Red Flag laps, and inaccurate timing records.
            - **Degradation-Preserving Outlier Filter:** First removes laps $>107\%$ of each driver's fastest lap in that race, then fits a local linear trend (`LapTime ~ TyreLife`) within each stint and removes residuals outside $\\pm 1.5 \\times \\text{IQR}$ (with a $0.15\\text{ s}$ minimum IQR floor) so genuinely degraded laps at the end of a long stint are preserved.
            """
        )
    with m_col2:
        st.markdown(
            """
            #### 2. Empirical Pit Loss & Fuel Load Calibration (`src/features.py`)
            - **Circuit Pit Stop Loss (No Fixed Constant):** Estimated per circuit from green-flag pit stops as:
              $$\\text{PitLoss}_c = \\text{median}\\big([\\text{InLap} + \\text{OutLap}] - 2 \\times \\text{DriverNormalLap}\\big)$$
              ranging from **`20.60 s` at Hungary** to **`24.24 s` at Netherlands** (and `26.77 s` at Monza in EDA).
            - **Fuel Weight Correction (`FuelAdjustedLapTime`):** An F1 car burns $\\approx 105\\text{ kg}$ of fuel from lights out to flag. Using multi-stint pit transitions where `TyreLife` resets while `LapNumber` advances, we calibrate $\\gamma_c \\in [0.0408, 0.0728]\\text{ s/lap}$ (`0.0280–0.0368 s/kg`):
              $$\\text{FuelAdjustedLapTime} = \\text{LapTimeSeconds} - \\gamma_c \\times (\\text{TotalRaceLaps} - \\text{LapNumber})$$
            - **Driver & Stint Normalization:** Normalizes baseline car speed (`Red Bull` vs. `Sauber`) so degradation models learn pure tyre wear rather than team budget.
            """
        )

    st.markdown("---")
    st.markdown("#### 3. Leak-Free 5-Fold Leave-One-Race-Out Cross-Validation (`src/model.py`)")
    st.caption(
        "Random train/test splitting across laps is prohibited because consecutive laps within a stint are autocorrelated. "
        "Instead, each fold holds out one entire Grand Prix as the unseen test set and trains strictly on the remaining four races."
    )

    c_v1, c_v2 = st.columns([1.3, 1.0])
    with c_v1:
        st.dataframe(loro_df, width="stretch", hide_index=True)
    with c_v2:
        tyre_err = err_df[err_df["SliceType"] == "TyreLifeBucket"].copy()
        fig_err = go.Figure()
        fig_err.add_trace(
            go.Bar(
                x=tyre_err["Regime"],
                y=tyre_err["Baseline_MAE_s"],
                name="Baseline Linear MAE",
                marker_color="#64748B",
                text=tyre_err["Baseline_MAE_s"].map(lambda v: f"{v:.3f}s"),
                textposition="outside",
            )
        )
        fig_err.add_trace(
            go.Bar(
                x=tyre_err["Regime"],
                y=tyre_err["Hybrid_MAE_s"],
                name="Hybrid Physics+GBDT MAE",
                marker_color="#FF2E4C",
                text=tyre_err["Hybrid_MAE_s"].map(lambda v: f"{v:.3f}s"),
                textposition="outside",
            )
        )
        fig_err.update_layout(barmode="group")
        fig_err.update_yaxes(title_text="Out-of-Fold MAE (seconds)", range=[0, 0.82])
        apply_broadcast_layout(
            fig_err,
            title="Hybrid Model Cuts Error by 19.6% in the Late-Stint Cliff (>25 Laps)",
            subtitle="5-Fold Leave-One-Race-Out MAE by tyre age bucket",
            height=360,
        )
        st.plotly_chart(fig_err, width="stretch")

    st.markdown("---")
    st.markdown(
        """
        #### 4. Honest Limitations & Operational Risks
        1. **Dirty Air & Midfield DRS Trains:** Pre-race strategy simulation cannot know in advance if a driver will emerge from the pit lane behind a 4-car midfield DRS train (which adds `+0.8 to +1.4 s/lap` of aerodynamic wake loss and thermal graining).
        2. **Safety Car / VSC Opportunism:** Our simulator optimizes green-flag race time. A Safety Car or Virtual Safety Car cuts effective pit stop loss by ~40%–55% (`~9–11 s` saved), which can instantly make an opportunistic 2-stop unbeatable even at a 1-stop circuit.
        3. **Late-Stint Thermal Cliff Variance (`>25 Laps`):** As shown in the error chart above, prediction MAE nearly doubles from `0.282 s` on fresh tyres (`1–5 laps`) to `0.555 s` beyond `25 laps` due to non-linear carcass wear and driver lift-and-coast management.
        4. **Relative Pirelli Compound Labels (`C1`–`C5`):** FastF1 reports session-relative labels (`SOFT`, `MEDIUM`, `HARD`) rather than Pirelli's absolute compound hardness (`C1` hardest to `C5` softest).

        ---
        *Disclaimer: This project uses **FastF1**, an unofficial open-source Python library, and is not associated in any way with the Formula 1 group of companies.*
        """
    )

    with st.expander("📂 View Full Data Audit & Fuel Calibration Tables"):
        st.markdown("**Race Data Quality Audit Table**")
        st.dataframe(audit_df, width="stretch", hide_index=True)
        st.markdown("**Empirical Fuel Correction Calibration Table**")
        st.dataframe(fuel_df, width="stretch", hide_index=True)


def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    bundle = load_dashboard_bundle()
    selected_circuit = render_sidebar(bundle)

    st.markdown(
        f"# 🏎️ F1 Tyre Degradation & Pit Strategy Optimizer"
    )
    st.markdown(
        f"**Active Circuit Focus:** `{selected_circuit} ({DEFAULT_RACES_2024[selected_circuit]})`  |  "
        "**Decision Question:** *When is the optimal pit stop window and which compound sequence minimizes total race time?*"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📌 1. Executive Summary",
            "📉 2. Tyre Degradation Curves",
            "🏁 3. Driver Strategy Timeline",
            "🎛️ 4. Interactive Pit Simulator",
            "🔬 5. Methodology & Limitations",
        ]
    )

    with tab1:
        render_tab_summary(bundle, selected_circuit)
    with tab2:
        render_tab_degradation(bundle, selected_circuit)
    with tab3:
        render_tab_timeline(bundle, selected_circuit)
    with tab4:
        render_tab_simulator(bundle, selected_circuit)
    with tab5:
        render_tab_methodology(bundle)


if __name__ == "__main__":
    main()
