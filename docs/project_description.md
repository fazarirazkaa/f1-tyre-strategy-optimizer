# Project Description — F1 Tyre Degradation & Pit Strategy Optimizer

**Word Count:** 248 words (Target: $\le 300$ words)

### Problem
In Formula 1, race outcomes are frequently decided on the pit wall rather than through on-track overtaking. Race strategists must continuously balance a high-stakes trade-off: losing `20.6 to 24.2 seconds` in the pit lane versus losing `0.04 to 0.12 seconds per lap` on degrading tyres. However, raw lap times are deceptive because cars burn `~105 kg` of fuel over a race (`0.041–0.073 s/lap` weight reduction), masking true tyre wear until a sudden late-stint thermal cliff occurs.

### Solution & Key Features
We developed the **F1 Tyre Degradation & Pit Strategy Optimizer**, an end-to-end decision-support pipeline and interactive **Streamlit + Plotly** dashboard built on `5,196` clean racing laps across five 2024 dry Grands Prix (**Japan, Spain, Hungary, Netherlands, and Abu Dhabi**):
- **Empirical Telemetry Calibration**: Estimates circuit-specific green-flag pit stop loss directly from `[in-lap + out-lap]` telemetry (`20.60 s` in Hungary to `24.24 s` in the Netherlands) and removes fuel-mass effects to isolate pure compound wear.
- **Leak-Free Machine Learning Validation**: A **Hybrid Physics + Monotonic HistGradientBoosting** model validated via **5-Fold Leave-One-Race-Out Cross-Validation (LORO-CV)** reduces out-of-race prediction error by **`9.76%`** overall (`0.4469 s` MAE vs. `0.4952 s` linear baseline) and by **`19.6%`** in the late-stint thermal cliff regime (`>25 laps`).
- **FIA-Compliant Strategy Simulator**: Exhaustively simulates 1-stop and 2-stop compound sequences enforcing **FIA Article 30.5(m)** (minimum two dry compounds) alongside a 2D sensitivity heatmap.

### Actionable Impact
Strategists can pinpoint exact pit windows before lights-out and adapt dynamically during the race—identifying when a **2-stop (`MEDIUM → SOFT → HARD`)** yields a decisive victory (`+6.70 s` in Hungary, `+1.09 s` in Spain) versus when track pit loss makes a **1-stop (`MEDIUM → HARD`)** unbeatable (`+6.52 s` in Abu Dhabi, `+4.79 s` in Japan, `+2.04 s` in the Netherlands).
