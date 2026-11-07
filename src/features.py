"""Feature engineering and empirical fuel-correction calibration for F1 race laps.

This module implements Stage 2 feature creation:
1. Empirical calibration of the fuel burn penalty per circuit (`FuelCorrectionPerLap`)
   leveraging multi-stint transitions where `LapNumber` advances while `TyreLife` resets,
   anchored to FIA regulatory fuel mass (110 kg max capacity) and F1 telemetry physics
   (~0.030-0.036 s/kg mass sensitivity).
2. Calculation of `FuelAdjustedLapTime` (normalizing all laps to an end-of-race empty-tank
   equivalent so true tyre wear is isolated from fuel burn-off).
3. Driver and circuit baseline pace normalization (Decision A3) so car performance gaps
   (e.g., Red Bull vs. Sauber) and circuit length differences do not confound tyre
   degradation slopes or Leave-One-Race-Out cross-validation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Physical & FIA regulatory constants (documented for reproducibility)
FIA_MAX_FUEL_LOAD_KG: float = 105.0  # Typical race start fuel load in modern F1 (kg, max allowed 110 kg)
BENCHMARK_MASS_SENSITIVITY_S_PER_KG_90S: float = 0.033  # Standard F1 sensitivity (~0.033 s/kg on a 90s lap)
MIN_VALID_S_PER_KG: float = 0.028  # Physical lower bound for mass sensitivity (s/kg)
MAX_VALID_S_PER_KG: float = 0.038  # Physical upper bound for mass sensitivity (s/kg)

COMPOUND_TO_CODE: Dict[str, int] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
}


def calibrate_fuel_correction(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate per-circuit fuel penalty (seconds/lap) from multi-stint race data.

    Methodology (Decision A2):
    In a race of `N` total laps, a car carries approximately `M_0 = 105 kg` of fuel at
    lights out and burns `M_0 / N` kg per lap. Within a single stint, `LapNumber` and
    `TyreLife` are perfectly collinear (`d(LapNumber) = d(TyreLife)`), but across pit
    stops `TyreLife` resets while `LapNumber` continues increasing.

    We estimate the fuel effect in two complementary ways:
    1. **Unconstrained Multi-Stint Fixed-Effects OLS**: For each circuit (and pooled), we
       regress `LapTimeSeconds` on driver fixed effects, compound fixed effects,
       compound-specific `TyreLife` slopes, and `RemainingFuelKg`.
    2. **Physics-Scaled Empirical Calibration**: Because compound sequence in F1 is not
       randomized (drivers often start on Soft/Medium and switch to Hard later, and track
       rubber evolution also reduces lap times late in the race), we scale the pooled
       empirical mass sensitivity by each circuit's median lap time relative to a 90s
       reference lap and clip to the physical F1 telemetry envelope `[0.028, 0.038] s/kg`.

    Returns
    -------
    pd.DataFrame
        Per-circuit fuel calibration table containing:
        - `Circuit`, `TotalRaceLaps`, `MedianRawLapSeconds`
        - `FuelBurnKgPerLap`: Estimated fuel consumed per lap (`105.0 / TotalRaceLaps`)
        - `UnconstrainedOLS_s_per_lap`: Raw multi-stint regression slope on `-LapNumber`
        - `Calibrated_s_per_kg`: Calibrated mass sensitivity (s/kg)
        - `FuelPenaltySecondsPerLap`: Final calibrated fuel effect in s/lap (`gamma_c`)
        - `LiteratureBenchmark_s_per_lap`: Reference F1 engineering value
    """
    rows: List[Dict[str, object]] = []

    # Step 1: Estimate raw multi-stint OLS slope per circuit controlling for Driver, Compound, TyreLife
    raw_s_per_kg_estimates: List[float] = []

    for circuit, grp in clean_df.groupby("Circuit", sort=False):
        n_laps = int(grp["TotalRaceLaps"].iloc[0])
        median_lap_s = float(grp["LapTimeSeconds"].median())
        fuel_burn_kg_per_lap = FIA_MAX_FUEL_LOAD_KG / float(n_laps)

        # Build design matrix for within-driver, within-compound multi-stint identification:
        # LapTimeSeconds ~ Driver + Compound + Compound:TyreLife + LapNumber
        df_c = grp[["LapTimeSeconds", "Driver", "Compound", "TyreLife", "LapNumber"]].dropna().copy()
        y = df_c["LapTimeSeconds"].to_numpy(dtype=float)

        X_parts = [
            pd.get_dummies(df_c["Driver"], prefix="drv", drop_first=True, dtype=float),
            pd.get_dummies(df_c["Compound"], prefix="cmp", drop_first=True, dtype=float),
        ]
        for comp in ("SOFT", "MEDIUM", "HARD"):
            mask_comp = (df_c["Compound"] == comp).astype(float)
            X_parts.append(pd.Series(mask_comp * df_c["TyreLife"], name=f"tyre_{comp}"))

        X_parts.append(df_c[["LapNumber"]].astype(float))
        X_df = pd.concat(X_parts, axis=1)
        X_df["intercept"] = 1.0

        X_mat = X_df.to_numpy(dtype=float)
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        lap_num_idx = list(X_df.columns).index("LapNumber")
        slope_lap_number = float(beta[lap_num_idx])  # Negative when car gets faster as fuel burns

        # Convert d(LapTime)/d(LapNumber) to positive fuel penalty per lap
        ols_s_per_lap = max(-slope_lap_number, 0.0)
        ols_s_per_kg = ols_s_per_lap / fuel_burn_kg_per_lap
        if MIN_VALID_S_PER_KG <= ols_s_per_kg <= MAX_VALID_S_PER_KG * 1.25:
            raw_s_per_kg_estimates.append(ols_s_per_kg)

        rows.append(
            {
                "Circuit": circuit,
                "TotalRaceLaps": n_laps,
                "MedianRawLapSeconds": round(median_lap_s, 3),
                "FuelBurnKgPerLap": round(fuel_burn_kg_per_lap, 4),
                "UnconstrainedOLS_s_per_lap": round(ols_s_per_lap, 4),
                "UnconstrainedOLS_s_per_kg": round(ols_s_per_kg, 4),
            }
        )

    calib_df = pd.DataFrame(rows)

    # Pooled empirical mass sensitivity anchored by valid multi-stint estimates & benchmark
    if len(raw_s_per_kg_estimates) > 0:
        base_s_per_kg = float(np.median(raw_s_per_kg_estimates))
        base_s_per_kg = float(np.clip(base_s_per_kg, MIN_VALID_S_PER_KG, MAX_VALID_S_PER_KG))
    else:
        base_s_per_kg = BENCHMARK_MASS_SENSITIVITY_S_PER_KG_90S

    # Scale mass sensitivity by circuit lap duration relative to a 90-second reference lap,
    # blended 50/50 with bounded circuit-specific OLS estimate for empirical fidelity
    calibrated_s_per_kg_list: List[float] = []
    calibrated_s_per_lap_list: List[float] = []
    benchmark_s_per_lap_list: List[float] = []

    for _, row in calib_df.iterrows():
        lap_ratio = float(row["MedianRawLapSeconds"]) / 90.0
        phys_s_per_kg = np.clip(
            base_s_per_kg * lap_ratio, MIN_VALID_S_PER_KG, MAX_VALID_S_PER_KG
        )
        circuit_ols_s_per_kg = np.clip(
            float(row["UnconstrainedOLS_s_per_kg"]), MIN_VALID_S_PER_KG, MAX_VALID_S_PER_KG
        )
        final_s_per_kg = float(0.5 * phys_s_per_kg + 0.5 * circuit_ols_s_per_kg)
        fuel_burn = float(row["FuelBurnKgPerLap"])

        final_s_per_lap = final_s_per_kg * fuel_burn
        bench_s_per_lap = (BENCHMARK_MASS_SENSITIVITY_S_PER_KG_90S * lap_ratio) * fuel_burn

        calibrated_s_per_kg_list.append(round(final_s_per_kg, 4))
        calibrated_s_per_lap_list.append(round(final_s_per_lap, 4))
        benchmark_s_per_lap_list.append(round(bench_s_per_lap, 4))

    calib_df["Calibrated_s_per_kg"] = calibrated_s_per_kg_list
    calib_df["FuelPenaltySecondsPerLap"] = calibrated_s_per_lap_list
    calib_df["LiteratureBenchmark_s_per_lap"] = benchmark_s_per_lap_list

    return calib_df


def build_features(
    clean_df: pd.DataFrame,
    fuel_calib_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create modelling and EDA features on cleaned representative laps.

    Features engineered:
    - `TyreLife`: Laps completed on current set of tyres (int).
    - `Compound`: Dry tyre compound (`SOFT`, `MEDIUM`, `HARD`).
    - `CompoundCode`: Ordinal encoding (`SOFT=0`, `MEDIUM=1`, `HARD=2`).
    - `Stint`: Stint number for that driver in the race (int).
    - `StintLength`: Total representative laps observed in that `(Circuit, Driver, Stint)`.
    - `LapNumber`: Race lap number (int).
    - `RemainingLaps`: `TotalRaceLaps - LapNumber`.
    - `RaceProgress`: `LapNumber / TotalRaceLaps` in `[0, 1]`.
    - `TrackTemp`, `AirTemp`, `TempDelta`: Track and ambient temperatures (deg C).
    - `FuelPenaltySeconds`: `FuelPenaltySecondsPerLap * (TotalRaceLaps - LapNumber)`.
    - `FuelAdjustedLapTime`: `LapTimeSeconds - FuelPenaltySeconds` (end-of-race zero-fuel basis).
    - `CircuitBaselinePace`: Median `FuelAdjustedLapTime` across all clean laps at that circuit.
    - `DriverBaselinePace`: Median `FuelAdjustedLapTime` of `(Circuit, Driver)` for early/mid
      tyre life (`TyreLife <= 12`, falling back to overall driver median if fewer than 3 laps),
      representing the fresh-tyre baseline pace of that driver/car at that circuit.
    - `DriverPaceOffset`: `DriverBaselinePace - CircuitBaselinePace` (car/driver speed advantage).
    - `FuelAdjustedDelta`: `FuelAdjustedLapTime - DriverBaselinePace` (pure degradation + thermal
      delta in seconds, invariant to circuit length and car speed).
    - `CircuitNormalizedLapTime`: `FuelAdjustedLapTime - DriverPaceOffset` (fuel-adjusted lap time
      normalized to the median car at that circuit).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - `feat_df`: Enriched DataFrame with all engineered features.
        - `fuel_calib_df`: Per-circuit fuel calibration table.
    """
    df = clean_df.copy()
    if fuel_calib_df is None:
        fuel_calib_df = calibrate_fuel_correction(df)

    # Merge fuel calibration parameters
    df = df.merge(
        fuel_calib_df[
            [
                "Circuit",
                "FuelBurnKgPerLap",
                "Calibrated_s_per_kg",
                "FuelPenaltySecondsPerLap",
            ]
        ],
        on="Circuit",
        how="left",
    )

    # Core integer/float features
    df["TyreLife"] = df["TyreLife"].astype(int)
    df["Stint"] = df["Stint"].astype(int)
    df["LapNumber"] = df["LapNumber"].astype(int)
    df["TotalRaceLaps"] = df["TotalRaceLaps"].astype(int)
    df["CompoundCode"] = df["Compound"].map(COMPOUND_TO_CODE).astype(int)

    # Race progress & remaining fuel features
    df["RemainingLaps"] = df["TotalRaceLaps"] - df["LapNumber"]
    df["RaceProgress"] = df["LapNumber"] / df["TotalRaceLaps"].astype(float)
    df["EstimatedFuelKg"] = df["RemainingLaps"] * df["FuelBurnKgPerLap"]

    # Temperature features (impute rare missing weather readings with circuit median)
    for col in ("TrackTemp", "AirTemp", "Humidity", "WindSpeed"):
        if col in df.columns:
            df[col] = df.groupby("Circuit")[col].transform(
                lambda s: s.astype(float).fillna(s.astype(float).median())
            )
    df["TempDelta"] = df["TrackTemp"] - df["AirTemp"]

    # Fuel-adjusted lap time (subtracting fuel weight penalty relative to empty tank at end of race)
    df["FuelPenaltySeconds"] = df["RemainingLaps"] * df["FuelPenaltySecondsPerLap"]
    df["FuelAdjustedLapTime"] = df["LapTimeSeconds"] - df["FuelPenaltySeconds"]

    # Stint length metadata (total clean laps & max tyre life reached in each driver stint)
    stint_stats = (
        df.groupby(["Circuit", "Driver", "Stint"])
        .agg(
            StintCleanLaps=("LapNumber", "count"),
            StintMaxTyreLife=("TyreLife", "max"),
            StintStartLap=("LapNumber", "min"),
            StintEndLap=("LapNumber", "max"),
        )
        .reset_index()
    )
    df = df.merge(stint_stats, on=["Circuit", "Driver", "Stint"], how="left")

    # Baseline pace normalization (Decision A3)
    # 1. Circuit baseline pace (median fuel-adjusted lap time across all drivers at that circuit)
    circuit_base = (
        df.groupby("Circuit")["FuelAdjustedLapTime"]
        .median()
        .rename("CircuitBaselinePace")
        .reset_index()
    )
    df = df.merge(circuit_base, on="Circuit", how="left")

    # 2. Driver baseline pace at each circuit (median of fresh-to-medium life laps TyreLife <= 12
    #    so one-stop vs two-stop strategy length doesn't bias a driver's baseline pace)
    fresh_subset = df[df["TyreLife"] <= 12]
    driver_fresh_counts = fresh_subset.groupby(["Circuit", "Driver"])["FuelAdjustedLapTime"].count()
    driver_fresh_medians = fresh_subset.groupby(["Circuit", "Driver"])["FuelAdjustedLapTime"].median()
    driver_all_medians = df.groupby(["Circuit", "Driver"])["FuelAdjustedLapTime"].median()

    driver_base_series = driver_all_medians.copy()
    valid_fresh_idx = driver_fresh_counts[driver_fresh_counts >= 3].index
    driver_base_series.loc[valid_fresh_idx] = driver_fresh_medians.loc[valid_fresh_idx]

    driver_base_df = driver_base_series.rename("DriverBaselinePace").reset_index()
    df = df.merge(driver_base_df, on=["Circuit", "Driver"], how="left")

    df["DriverPaceOffset"] = df["DriverBaselinePace"] - df["CircuitBaselinePace"]
    df["FuelAdjustedDelta"] = df["FuelAdjustedLapTime"] - df["DriverBaselinePace"]
    df["CircuitNormalizedLapTime"] = df["FuelAdjustedLapTime"] - df["DriverPaceOffset"]

    # 3. Stint-level fresh-tyre baseline pace (20th percentile of first 6 laps of the stint)
    #    Removes cross-stint track rubbering-in and compound base-speed offsets to isolate within-stint wear
    df["StintMinTyreLife"] = df.groupby(["Circuit", "Driver", "Stint"])["TyreLife"].transform("min")
    df["StintLapAge"] = df["TyreLife"] - df["StintMinTyreLife"]
    early_stint_mask = df["TyreLife"] <= (df["StintMinTyreLife"] + 5)
    early_stint_base = (
        df[early_stint_mask]
        .groupby(["Circuit", "Driver", "Stint"])["FuelAdjustedLapTime"]
        .quantile(0.20)
        .rename("StintBaselinePace")
        .reset_index()
    )
    df = df.merge(early_stint_base, on=["Circuit", "Driver", "Stint"], how="left")
    df["StintBaselinePace"] = df["StintBaselinePace"].fillna(df["DriverBaselinePace"])
    df["StintDegradationDelta"] = df["FuelAdjustedLapTime"] - df["StintBaselinePace"]

    # Keep only columns needed for downstream EDA, modeling, and simulation in a clean order
    selected_cols = [
        "Season",
        "Circuit",
        "EventName",
        "TotalRaceLaps",
        "Driver",
        "DriverNumber",
        "Team",
        "Position",
        "Stint",
        "Compound",
        "CompoundCode",
        "FreshTyre",
        "LapNumber",
        "RemainingLaps",
        "RaceProgress",
        "TyreLife",
        "StintMinTyreLife",
        "StintLapAge",
        "StintCleanLaps",
        "StintMaxTyreLife",
        "StintStartLap",
        "StintEndLap",
        "TrackTemp",
        "AirTemp",
        "TempDelta",
        "Humidity",
        "WindSpeed",
        "LapTimeSeconds",
        "EstimatedFuelKg",
        "FuelPenaltySecondsPerLap",
        "FuelPenaltySeconds",
        "FuelAdjustedLapTime",
        "CircuitBaselinePace",
        "DriverBaselinePace",
        "DriverPaceOffset",
        "FuelAdjustedDelta",
        "StintBaselinePace",
        "StintDegradationDelta",
        "CircuitNormalizedLapTime",
        "Sector1Seconds",
        "Sector2Seconds",
        "Sector3Seconds",
    ]
    existing_cols = [c for c in selected_cols if c in df.columns]
    feat_df = df[existing_cols].sort_values(["Circuit", "Driver", "LapNumber"]).reset_index(drop=True)

    return feat_df, fuel_calib_df
