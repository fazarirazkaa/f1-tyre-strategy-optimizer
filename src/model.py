"""Stage 3: Tyre degradation and lap time prediction models with Leave-One-Race-Out CV.

This module implements:
1. Baseline Linear Regression per compound and circuit (degradation slope in s/lap),
   including hierarchical empirical Bayes regularization for sparse regimes (e.g., Hungary Soft).
2. Expressive models predicting fuel-adjusted degradation & lap time from tyre life,
   compound, track/air temperature, and fuel load:
   - `BaselineLinearDegradationModel`: Per-compound OLS on `TyreLife`.
   - `RidgeMixedInteractionModel`: Physics-scaled $L_2$-regularized linear mixed-interaction model.
   - `MonotonicBoostingDegradationModel`: `HistGradientBoostingRegressor` with monotonic
     positive wear constraints on tyre age features.
   - `HybridPhysicsBoostingModel`: 50/50 ensemble uniting smooth linear physics with
     non-linear monotonic tree boosting.
3. Leak-free 5-Fold Leave-One-Race-Out Cross-Validation (LORO-CV) where each fold holds
   out one entire Grand Prix as the unseen test set.
4. Comprehensive error analysis across tyre age regimes, race traffic phases, compounds,
   and circuits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.load import DRY_COMPOUNDS, PROCESSED_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

REFERENCE_LAP_SECONDS: float = 85.0  # Reference F1 lap duration for cross-circuit scaling
MIN_PHYSICAL_SLOPE_S_PER_LAP: float = 0.025  # Physical lower floor for dry tyre wear (s/lap)

# Calibrated fresh-tyre pace offsets at TyreLife=1 relative to MEDIUM on an 85s lap
# (Derived from multi-stint fixed-effects regression excluding Lap 2-4 traffic pack)
DEFAULT_FRESH_COMPOUND_OFFSETS_85S: Dict[str, float] = {
    "SOFT": -0.235,
    "MEDIUM": 0.000,
    "HARD": +0.187,
}


def prepare_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-level modeling features without cross-row or cross-race leakage.

    All transformations depend strictly on single-lap state (`TyreLife`, `StintLapAge`,
    `Compound`, `FreshTyre`, `TrackTemp`, `AirTemp`, `EstimatedFuelKg`, `RaceProgress`)
    and circuit lap duration scale (`CircuitBaselinePace / 85.0`).
    """
    out = df.copy()

    if "StintMinTyreLife" not in out.columns:
        out["StintMinTyreLife"] = out.groupby(["Circuit", "Driver", "Stint"])["TyreLife"].transform("min")
    if "StintLapAge" not in out.columns:
        out["StintLapAge"] = out["TyreLife"] - out["StintMinTyreLife"]

    # Scale tyre age by circuit lap duration relative to an 85s reference lap
    # (A tyre spending 97s per lap at Suzuka experiences ~28% more load time per lap than a 76s lap at Zandvoort)
    out["LapDurationScale"] = out["CircuitBaselinePace"] / REFERENCE_LAP_SECONDS
    out["ScaledStintAge"] = out["StintLapAge"] * out["LapDurationScale"]
    out["ScaledTyreLife"] = out["TyreLife"] * out["LapDurationScale"]
    out["PriorTyreAge"] = (out["StintMinTyreLife"] - 1).clip(lower=0).astype(float)

    out["IsSoft"] = (out["Compound"] == "SOFT").astype(float)
    out["IsMedium"] = (out["Compound"] == "MEDIUM").astype(float)
    out["IsHard"] = (out["Compound"] == "HARD").astype(float)
    out["FreshTyreInt"] = out["FreshTyre"].fillna(True).astype(float)

    # Compound-specific wear interaction terms
    out["SoftStintAge"] = out["IsSoft"] * out["ScaledStintAge"]
    out["MediumStintAge"] = out["IsMedium"] * out["ScaledStintAge"]
    out["HardStintAge"] = out["IsHard"] * out["ScaledStintAge"]

    # Fuel & thermal interaction features
    out["FuelLoadRatio"] = out["EstimatedFuelKg"] / 105.0
    out["FuelAgeInteraction"] = out["ScaledStintAge"] * out["FuelLoadRatio"]
    out["TempDelta"] = out["TrackTemp"] - out["AirTemp"]
    out["ThermalAgeInteraction"] = out["ScaledStintAge"] * ((out["TrackTemp"] - 30.0) / 30.0)
    out["IsOpeningPack"] = (out["LapNumber"] <= 5).astype(float)

    return out


def estimate_compound_initial_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate initial fresh-tyre compound speed offsets (at TyreLife=1) per circuit.

    Excludes opening-pack traffic (`LapNumber >= 5`) and controls for `TyreLife` per
    compound and `RaceProgress` (track evolution) so the intercepts reflect the true
    fresh-tyre speed gap between `SOFT`, `MEDIUM`, and `HARD`.
    """
    sub = df[df["LapNumber"] >= 5].copy()
    X = pd.DataFrame(
        {
            "is_soft": (sub["Compound"] == "SOFT").astype(float),
            "is_hard": (sub["Compound"] == "HARD").astype(float),
            "tl_soft": (sub["Compound"] == "SOFT").astype(float) * sub["TyreLife"],
            "tl_med": (sub["Compound"] == "MEDIUM").astype(float) * sub["TyreLife"],
            "tl_hard": (sub["Compound"] == "HARD").astype(float) * sub["TyreLife"],
            "progress": sub["RaceProgress"].astype(float),
            "intercept": 1.0,
        }
    )
    beta, _, _, _ = np.linalg.lstsq(
        X.to_numpy(dtype=float), sub["FuelAdjustedDelta"].to_numpy(dtype=float), rcond=None
    )
    coef_map = dict(zip(X.columns, beta))
    soft_offset_85 = float(coef_map.get("is_soft", DEFAULT_FRESH_COMPOUND_OFFSETS_85S["SOFT"]))
    hard_offset_85 = float(coef_map.get("is_hard", DEFAULT_FRESH_COMPOUND_OFFSETS_85S["HARD"]))

    # Bound within physically realistic Pirelli C-range step gaps
    soft_offset_85 = float(np.clip(soft_offset_85, -0.45, -0.15))
    hard_offset_85 = float(np.clip(hard_offset_85, +0.12, +0.40))

    rows: List[Dict[str, object]] = []
    for circuit, grp in df.groupby("Circuit", sort=False):
        scale = float(grp["CircuitBaselinePace"].iloc[0]) / REFERENCE_LAP_SECONDS
        for comp, base_off in [
            ("SOFT", soft_offset_85),
            ("MEDIUM", 0.0),
            ("HARD", hard_offset_85),
        ]:
            rows.append(
                {
                    "Circuit": circuit,
                    "Compound": comp,
                    "FreshOffsetVsMediumSeconds": round(base_off * scale, 3),
                    "GlobalBaseOffset85s": round(base_off, 3),
                }
            )
    return pd.DataFrame(rows)


def fit_circuit_compound_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fit descriptive Baseline Linear Regression per circuit and compound.

    Computes both:
    - Raw OLS slope of `StintDegradationDelta ~ TyreLife`
    - Within-stint Fixed-Effects OLS slope (`FuelAdjustedLapTime ~ TyreLife + Driver:Stint`)
    - Regularized physical slope (applying empirical Bayes pooling toward the global
      compound slope when sample size `CleanLaps <= 35` and opening-lap traffic causes
      an unphysical negative raw slope, as documented in EDA for `(Hungary, SOFT)`).
    """
    feat_df = prepare_modeling_features(df)
    offsets_df = estimate_compound_initial_offsets(feat_df)

    # Global pooled slope per compound (excluding Hungary Soft outlier regime)
    global_comp_slopes: Dict[str, float] = {}
    for comp in DRY_COMPOUNDS:
        sub_c = feat_df[feat_df["Compound"] == comp]
        if comp == "SOFT":
            sub_c = sub_c[~((sub_c["Circuit"] == "Hungary") & (sub_c["LapNumber"] <= 6))]
        s_glob, _ = np.polyfit(
            sub_c["TyreLife"].to_numpy(dtype=float),
            sub_c["StintDegradationDelta"].to_numpy(dtype=float),
            1,
        )
        global_comp_slopes[comp] = max(float(s_glob), MIN_PHYSICAL_SLOPE_S_PER_LAP)

    rows: List[Dict[str, object]] = []
    for (circuit, comp), grp in feat_df.groupby(["Circuit", "Compound"], sort=False):
        n_laps = len(grp)
        n_stints = len(grp[["Driver", "Stint"]].drop_duplicates())
        x = grp["TyreLife"].to_numpy(dtype=float)
        y = grp["StintDegradationDelta"].to_numpy(dtype=float)

        raw_slope, raw_intercept = np.polyfit(x, y, 1)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0

        # Within-stint fixed-effects demeaned slope
        x_dm = x - grp.groupby(["Driver", "Stint"])["TyreLife"].transform("mean").to_numpy(dtype=float)
        y_dm = (
            grp["FuelAdjustedLapTime"].to_numpy(dtype=float)
            - grp.groupby(["Driver", "Stint"])["FuelAdjustedLapTime"].transform("mean").to_numpy(dtype=float)
        )
        denom = float(np.sum(x_dm ** 2))
        fe_slope = float(np.sum(x_dm * y_dm) / denom) if denom > 1e-6 else float(raw_slope)

        # Empirical Bayes shrinkage for small-sample traffic-confounded regime
        scale = float(grp["LapDurationScale"].iloc[0])
        prior_slope = global_comp_slopes[comp] * scale
        if n_laps <= 35 or raw_slope < MIN_PHYSICAL_SLOPE_S_PER_LAP:
            weight_data = max(0.0, (n_laps - 15.0) / 60.0) if raw_slope > 0 else 0.0
            reg_slope = weight_data * float(raw_slope) + (1.0 - weight_data) * prior_slope
            reg_intercept = float(np.median(y - reg_slope * (x - np.min(x))))
            is_shrunk = True
        else:
            reg_slope = float(raw_slope)
            reg_intercept = float(raw_intercept)
            is_shrunk = False

        y_pred_raw = raw_slope * x + raw_intercept
        in_mae = float(mean_absolute_error(y, y_pred_raw))
        in_rmse = float(root_mean_squared_error(y, y_pred_raw))
        in_r2 = float(r2_score(y, y_pred_raw))

        rows.append(
            {
                "Circuit": circuit,
                "Compound": comp,
                "Stints": int(n_stints),
                "CleanLaps": int(n_laps),
                "RawOLS_Slope_s_per_lap": round(float(raw_slope), 4),
                "WithinStintFE_Slope_s_per_lap": round(float(fe_slope), 4),
                "Regularized_Slope_s_per_lap": round(float(reg_slope), 4),
                "Intercept_s": round(float(reg_intercept), 3),
                "Correlation_r": round(corr, 3),
                "InSample_R2": round(in_r2, 3),
                "InSample_MAE_s": round(in_mae, 4),
                "InSample_RMSE_s": round(in_rmse, 4),
                "UsedEmpiricalBayesPrior": is_shrunk,
                "MeanTrackTemp_C": round(float(grp["TrackTemp"].mean()), 1),
            }
        )

    table_df = pd.DataFrame(rows)
    table_df = table_df.merge(
        offsets_df[["Circuit", "Compound", "FreshOffsetVsMediumSeconds"]],
        on=["Circuit", "Compound"],
        how="left",
    )
    return table_df


class BaselineLinearDegradationModel(BaseEstimator, RegressorMixin):
    """Baseline linear regression per tyre compound (`StintDegradationDelta ~ TyreLife`).

    In Leave-One-Race-Out CV, fits one OLS line (`slope * TyreLife + intercept`) per
    dry compound (`SOFT`, `MEDIUM`, `HARD`) exclusively on the training circuits and
    predicts on the held-out test circuit.
    """

    def __init__(self) -> None:
        self.models_: Dict[str, LinearRegression] = {}
        self.slopes_: Dict[str, float] = {}
        self.intercepts_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "BaselineLinearDegradationModel":
        y_arr = np.asarray(y, dtype=float)
        self.models_ = {}
        self.slopes_ = {}
        self.intercepts_ = {}

        for comp in DRY_COMPOUNDS:
            mask = (X["Compound"] == comp).to_numpy()
            if mask.sum() >= 2:
                lr = LinearRegression()
                lr.fit(X.loc[mask, ["TyreLife"]], y_arr[mask])
                self.models_[comp] = lr
                self.slopes_[comp] = float(lr.coef_[0])
                self.intercepts_[comp] = float(lr.intercept_)
            else:
                lr = LinearRegression()
                lr.fit(X[["TyreLife"]], y_arr)
                self.models_[comp] = lr
                self.slopes_[comp] = float(lr.coef_[0])
                self.intercepts_[comp] = float(lr.intercept_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X), dtype=float)
        for comp, lr in self.models_.items():
            mask = (X["Compound"] == comp).to_numpy()
            if mask.sum() > 0:
                preds[mask] = lr.predict(X.loc[mask, ["TyreLife"]])
        return preds


class RidgeMixedInteractionModel(BaseEstimator, RegressorMixin):
    """Physics-scaled $L_2$-regularized linear mixed-interaction model.

    Captures compound-specific wear slopes, circuit lap-duration scaling, prior tyre age
    at stint start, remaining fuel load interactions, and track-air temperature differentials.
    """

    FEATURE_COLS: List[str] = [
        "ScaledStintAge",
        "ScaledTyreLife",
        "PriorTyreAge",
        "IsSoft",
        "IsMedium",
        "FreshTyreInt",
        "SoftStintAge",
        "MediumStintAge",
        "HardStintAge",
        "FuelLoadRatio",
        "RaceProgress",
        "FuelAgeInteraction",
        "TempDelta",
    ]

    def __init__(self) -> None:
        self.pipeline_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "RidgeMixedInteractionModel":
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(0, 4, 30)),
        )
        self.pipeline_.fit(X[self.FEATURE_COLS], np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("RidgeMixedInteractionModel is not fitted yet.")
        return np.asarray(self.pipeline_.predict(X[self.FEATURE_COLS]), dtype=float)


class MonotonicBoostingDegradationModel(BaseEstimator, RegressorMixin):
    """Expressive Gradient Boosting model with physical monotonic wear constraints.

    Uses `HistGradientBoostingRegressor` with `loss='absolute_error'` (robust to traffic
    spikes) and enforces positive monotonic constraints (`+1`) on all tyre age features
    so predicted lap time never unphysically decreases as tyres age.
    """

    FEATURE_COLS: List[str] = [
        "ScaledStintAge",
        "ScaledTyreLife",
        "CompoundCode",
        "FreshTyreInt",
        "PriorTyreAge",
        "FuelLoadRatio",
        "RaceProgress",
        "SoftStintAge",
        "MediumStintAge",
        "HardStintAge",
        "FuelAgeInteraction",
        "TempDelta",
    ]
    # +1: non-decreasing with tyre wear; -1: SOFT(0) >= MEDIUM(1) >= HARD(2) wear rate; 0: unconstrained
    MONOTONIC_CST: List[int] = [1, 1, -1, 0, 1, 0, 0, 1, 1, 1, 1, 0]

    def __init__(
        self,
        max_iter: int = 120,
        max_depth: int = 4,
        learning_rate: float = 0.04,
        min_samples_leaf: int = 40,
        l2_regularization: float = 15.0,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.model_: Optional[HistGradientBoostingRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "MonotonicBoostingDegradationModel":
        self.model_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            monotonic_cst=self.MONOTONIC_CST,
            random_state=self.random_state,
        )
        self.model_.fit(X[self.FEATURE_COLS], np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MonotonicBoostingDegradationModel is not fitted yet.")
        return np.asarray(self.model_.predict(X[self.FEATURE_COLS]), dtype=float)


class HybridPhysicsBoostingModel(BaseEstimator, RegressorMixin):
    """Main expressive model: 50/50 ensemble of Ridge Mixed-Interaction + Monotonic HGB.

    Why this architecture excels in Leave-One-Race-Out CV:
    Tree-based models capture non-linear tyre wear cliffs and fuel-thermal interactions
    but produce piecewise-constant step functions. Blending `MonotonicBoostingDegradationModel`
    with `RidgeMixedInteractionModel` preserves smooth continuous derivatives with respect to
    `TyreLife` (essential for finding exact optimal pit stop windows in Stage 4) while
    capturing non-linear degradation curvature.
    """

    def __init__(self, ridge_weight: float = 0.5) -> None:
        self.ridge_weight = ridge_weight
        self.ridge_ = RidgeMixedInteractionModel()
        self.hgb_ = MonotonicBoostingDegradationModel()

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "HybridPhysicsBoostingModel":
        self.ridge_.fit(X, y)
        self.hgb_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_r = self.ridge_.predict(X)
        pred_h = self.hgb_.predict(X)
        return self.ridge_weight * pred_r + (1.0 - self.ridge_weight) * pred_h


def run_leave_one_race_out_cv(
    df: pd.DataFrame,
    target_col: str = "StintDegradationDelta",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute leak-free 5-fold Leave-One-Race-Out Cross-Validation (LORO-CV).

    In each fold `k`, one entire Grand Prix (`Circuit == test_circuit`) is held out as
    the test fold, and all models are fitted from scratch on the remaining 4 Grand Prix.
    Random splitting across laps is strictly prohibited to prevent intra-stint leakage.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - `metrics_df`: Per-fold and summary Mean/Std MAE & RMSE (seconds) for all models.
        - `oof_df`: Out-of-fold predictions and reconstructed lap times (`LapTimeSeconds`)
          for every clean lap in the dataset.
    """
    feat_df = prepare_modeling_features(df)
    circuits = list(feat_df["Circuit"].unique())

    oof_frames: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, test_circuit in enumerate(circuits, start=1):
        train_df = feat_df[feat_df["Circuit"] != test_circuit].copy()
        test_df = feat_df[feat_df["Circuit"] == test_circuit].copy()

        y_train = train_df[target_col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)

        # Fit models strictly on the 4 training races
        m_base = BaselineLinearDegradationModel().fit(train_df, y_train)
        m_ridge = RidgeMixedInteractionModel().fit(train_df, y_train)
        m_hgb = MonotonicBoostingDegradationModel().fit(train_df, y_train)
        m_hybrid = HybridPhysicsBoostingModel(ridge_weight=0.5).fit(train_df, y_train)

        # Predict on the held-out 5th race
        p_base = m_base.predict(test_df)
        p_ridge = m_ridge.predict(test_df)
        p_hgb = m_hgb.predict(test_df)
        p_hybrid = m_hybrid.predict(test_df)

        # Reconstruct full LapTimeSeconds: StintBaselinePace + FuelPenaltySeconds + pred_delta
        base_pace_plus_fuel = (
            test_df["StintBaselinePace"].to_numpy(dtype=float)
            + test_df["FuelPenaltySeconds"].to_numpy(dtype=float)
        )
        test_out = test_df.copy()
        test_out["Fold"] = fold_idx
        test_out["Pred_Baseline_Delta"] = p_base
        test_out["Pred_Ridge_Delta"] = p_ridge
        test_out["Pred_MonotonicHGB_Delta"] = p_hgb
        test_out["Pred_Hybrid_Delta"] = p_hybrid

        test_out["Pred_Baseline_LapTime"] = base_pace_plus_fuel + p_base
        test_out["Pred_Ridge_LapTime"] = base_pace_plus_fuel + p_ridge
        test_out["Pred_MonotonicHGB_LapTime"] = base_pace_plus_fuel + p_hgb
        test_out["Pred_Hybrid_LapTime"] = base_pace_plus_fuel + p_hybrid

        # Errors on LapTimeSeconds are identical to errors on StintDegradationDelta
        test_out["Error_Baseline"] = test_out["LapTimeSeconds"] - test_out["Pred_Baseline_LapTime"]
        test_out["Error_Ridge"] = test_out["LapTimeSeconds"] - test_out["Pred_Ridge_LapTime"]
        test_out["Error_MonotonicHGB"] = test_out["LapTimeSeconds"] - test_out["Pred_MonotonicHGB_LapTime"]
        test_out["Error_Hybrid"] = test_out["LapTimeSeconds"] - test_out["Pred_Hybrid_LapTime"]

        test_out["AbsError_Baseline"] = np.abs(test_out["Error_Baseline"])
        test_out["AbsError_Ridge"] = np.abs(test_out["Error_Ridge"])
        test_out["AbsError_MonotonicHGB"] = np.abs(test_out["Error_MonotonicHGB"])
        test_out["AbsError_Hybrid"] = np.abs(test_out["Error_Hybrid"])

        oof_frames.append(test_out)

        base_mae = float(mean_absolute_error(y_test, p_base))
        base_rmse = float(root_mean_squared_error(y_test, p_base))
        ridge_mae = float(mean_absolute_error(y_test, p_ridge))
        ridge_rmse = float(root_mean_squared_error(y_test, p_ridge))
        hgb_mae = float(mean_absolute_error(y_test, p_hgb))
        hgb_rmse = float(root_mean_squared_error(y_test, p_hgb))
        hyb_mae = float(mean_absolute_error(y_test, p_hybrid))
        hyb_rmse = float(root_mean_squared_error(y_test, p_hybrid))

        fold_rows.append(
            {
                "Fold": f"Fold {fold_idx}",
                "TestCircuit": test_circuit,
                "TestLaps": int(len(test_df)),
                "Baseline_MAE_s": round(base_mae, 4),
                "Baseline_RMSE_s": round(base_rmse, 4),
                "Ridge_MAE_s": round(ridge_mae, 4),
                "Ridge_RMSE_s": round(ridge_rmse, 4),
                "MonotonicHGB_MAE_s": round(hgb_mae, 4),
                "MonotonicHGB_RMSE_s": round(hgb_rmse, 4),
                "Hybrid_MAE_s": round(hyb_mae, 4),
                "Hybrid_RMSE_s": round(hyb_rmse, 4),
                "MAE_Improvement_vs_Baseline_Pct": round(
                    100.0 * (base_mae - hyb_mae) / base_mae, 2
                ),
                "RMSE_Improvement_vs_Baseline_Pct": round(
                    100.0 * (base_rmse - hyb_rmse) / base_rmse, 2
                ),
            }
        )

    oof_df = pd.concat(oof_frames, ignore_index=True)
    folds_only_df = pd.DataFrame(fold_rows)

    # Append Mean and Std summary rows across the 5 folds
    mean_base_mae = float(folds_only_df["Baseline_MAE_s"].mean())
    mean_base_rmse = float(folds_only_df["Baseline_RMSE_s"].mean())
    mean_hyb_mae = float(folds_only_df["Hybrid_MAE_s"].mean())
    mean_hyb_rmse = float(folds_only_df["Hybrid_RMSE_s"].mean())

    summary_rows = [
        {
            "Fold": "Mean (5-Fold LORO)",
            "TestCircuit": "ALL 5 RACES",
            "TestLaps": int(folds_only_df["TestLaps"].sum()),
            "Baseline_MAE_s": round(mean_base_mae, 4),
            "Baseline_RMSE_s": round(mean_base_rmse, 4),
            "Ridge_MAE_s": round(float(folds_only_df["Ridge_MAE_s"].mean()), 4),
            "Ridge_RMSE_s": round(float(folds_only_df["Ridge_RMSE_s"].mean()), 4),
            "MonotonicHGB_MAE_s": round(float(folds_only_df["MonotonicHGB_MAE_s"].mean()), 4),
            "MonotonicHGB_RMSE_s": round(float(folds_only_df["MonotonicHGB_RMSE_s"].mean()), 4),
            "Hybrid_MAE_s": round(mean_hyb_mae, 4),
            "Hybrid_RMSE_s": round(mean_hyb_rmse, 4),
            "MAE_Improvement_vs_Baseline_Pct": round(
                100.0 * (mean_base_mae - mean_hyb_mae) / mean_base_mae, 2
            ),
            "RMSE_Improvement_vs_Baseline_Pct": round(
                100.0 * (mean_base_rmse - mean_hyb_rmse) / mean_base_rmse, 2
            ),
        },
        {
            "Fold": "Std Dev (5-Fold LORO)",
            "TestCircuit": "STD ACROSS FOLDS",
            "TestLaps": 0,
            "Baseline_MAE_s": round(float(folds_only_df["Baseline_MAE_s"].std(ddof=1)), 4),
            "Baseline_RMSE_s": round(float(folds_only_df["Baseline_RMSE_s"].std(ddof=1)), 4),
            "Ridge_MAE_s": round(float(folds_only_df["Ridge_MAE_s"].std(ddof=1)), 4),
            "Ridge_RMSE_s": round(float(folds_only_df["Ridge_RMSE_s"].std(ddof=1)), 4),
            "MonotonicHGB_MAE_s": round(float(folds_only_df["MonotonicHGB_MAE_s"].std(ddof=1)), 4),
            "MonotonicHGB_RMSE_s": round(float(folds_only_df["MonotonicHGB_RMSE_s"].std(ddof=1)), 4),
            "Hybrid_MAE_s": round(float(folds_only_df["Hybrid_MAE_s"].std(ddof=1)), 4),
            "Hybrid_RMSE_s": round(float(folds_only_df["Hybrid_RMSE_s"].std(ddof=1)), 4),
            "MAE_Improvement_vs_Baseline_Pct": round(
                float(folds_only_df["MAE_Improvement_vs_Baseline_Pct"].std(ddof=1)), 2
            ),
            "RMSE_Improvement_vs_Baseline_Pct": round(
                float(folds_only_df["RMSE_Improvement_vs_Baseline_Pct"].std(ddof=1)), 2
            ),
        },
    ]

    metrics_df = pd.concat([folds_only_df, pd.DataFrame(summary_rows)], ignore_index=True)
    return metrics_df, oof_df


def analyze_model_errors(oof_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Slice out-of-fold LORO-CV residuals to diagnose regimes where the model is weakest.

    Slices analyzed:
    1. `by_tyre_life`: Fresh (1-5), Mid-Stint (6-15), Worn (16-25), Cliff (>25 laps)
    2. `by_race_phase`: Opening Pack (Laps 2-5), Mid-Race (Laps 6-40), Late-Race (>40)
    3. `by_compound`: SOFT, MEDIUM, HARD
    4. `worst_stints`: Top 10 individual driver stints with highest MAE and root-cause notes
    """
    df = oof_df.copy()

    df["TyreLifeBucket"] = pd.cut(
        df["TyreLife"],
        bins=[0, 5, 15, 25, 100],
        labels=[
            "1-5 Laps (Fresh / Warm-up)",
            "6-15 Laps (Prime Stint Window)",
            "16-25 Laps (High Wear Window)",
            ">25 Laps (Extended / Thermal Cliff)",
        ],
    )
    df["RacePhaseBucket"] = pd.cut(
        df["LapNumber"],
        bins=[0, 5, 40, 100],
        labels=[
            "Laps 2-5 (Opening Pack / DRS Train)",
            "Laps 6-40 (Mid-Race Strategic Phase)",
            "Laps >40 (Late-Race Low Fuel)",
        ],
    )

    def _summarize_slice(group_col: str) -> pd.DataFrame:
        rows = []
        for val, grp in df.groupby(group_col, observed=False):
            if len(grp) == 0:
                continue
            b_mae = float(mean_absolute_error(grp["LapTimeSeconds"], grp["Pred_Baseline_LapTime"]))
            h_mae = float(mean_absolute_error(grp["LapTimeSeconds"], grp["Pred_Hybrid_LapTime"]))
            b_rmse = float(root_mean_squared_error(grp["LapTimeSeconds"], grp["Pred_Baseline_LapTime"]))
            h_rmse = float(root_mean_squared_error(grp["LapTimeSeconds"], grp["Pred_Hybrid_LapTime"]))
            bias = float(np.mean(grp["Error_Hybrid"]))  # Positive = actual slower than predicted
            rows.append(
                {
                    "SliceType": group_col,
                    "Regime": str(val),
                    "Laps": int(len(grp)),
                    "Baseline_MAE_s": round(b_mae, 4),
                    "Hybrid_MAE_s": round(h_mae, 4),
                    "Baseline_RMSE_s": round(b_rmse, 4),
                    "Hybrid_RMSE_s": round(h_rmse, 4),
                    "Hybrid_SignedBias_s": round(bias, 4),
                }
            )
        return pd.DataFrame(rows)

    by_tyre_life = _summarize_slice("TyreLifeBucket")
    by_race_phase = _summarize_slice("RacePhaseBucket")
    by_compound = _summarize_slice("Compound")

    combined_regimes = pd.concat([by_tyre_life, by_race_phase, by_compound], ignore_index=True)

    # Identify worst-predicted individual driver stints (>= 5 clean laps)
    stint_errors = (
        df.groupby(["Circuit", "Driver", "Team", "Stint", "Compound"], as_index=False)
        .agg(
            CleanLaps=("LapNumber", "count"),
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            MaxTyreLife=("TyreLife", "max"),
            Baseline_MAE_s=("AbsError_Baseline", "mean"),
            Hybrid_MAE_s=("AbsError_Hybrid", "mean"),
            MeanSignedError_s=("Error_Hybrid", "mean"),
        )
    )
    stint_errors = (
        stint_errors[stint_errors["CleanLaps"] >= 5]
        .sort_values("Hybrid_MAE_s", ascending=False)
        .head(10)
        .round(3)
        .reset_index(drop=True)
    )

    return {
        "combined_regimes": combined_regimes,
        "by_tyre_life": by_tyre_life,
        "by_race_phase": by_race_phase,
        "by_compound": by_compound,
        "worst_stints": stint_errors,
    }


def fit_full_models(df: pd.DataFrame) -> Dict[str, object]:
    """Fit Baseline, Ridge, Monotonic HGB, and Hybrid models on all 5 races for Stage 4/5."""
    feat_df = prepare_modeling_features(df)
    y = feat_df["StintDegradationDelta"].to_numpy(dtype=float)

    m_base = BaselineLinearDegradationModel().fit(feat_df, y)
    m_ridge = RidgeMixedInteractionModel().fit(feat_df, y)
    m_hgb = MonotonicBoostingDegradationModel().fit(feat_df, y)
    m_hybrid = HybridPhysicsBoostingModel(ridge_weight=0.5).fit(feat_df, y)

    return {
        "baseline": m_base,
        "ridge": m_ridge,
        "monotonic_hgb": m_hgb,
        "hybrid": m_hybrid,
        "baseline_table": fit_circuit_compound_baseline_table(feat_df),
        "compound_offsets": estimate_compound_initial_offsets(feat_df),
    }


def run_stage3_pipeline(save_outputs: bool = True) -> Dict[str, pd.DataFrame]:
    """Execute complete Stage 3 modeling, LORO-CV validation, and error analysis."""
    clean_path = PROCESSED_DIR / "clean_laps_2024.csv"
    if not clean_path.exists():
        from src.load import run_stage2_pipeline

        run_stage2_pipeline(save_processed=True)

    df = pd.read_csv(clean_path)
    baseline_table = fit_circuit_compound_baseline_table(df)
    compound_offsets = estimate_compound_initial_offsets(df)
    loro_metrics, oof_df = run_leave_one_race_out_cv(df)
    error_dict = analyze_model_errors(oof_df)

    if save_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        baseline_table.to_csv(PROCESSED_DIR / "baseline_circuit_compound_slopes_2024.csv", index=False)
        compound_offsets.to_csv(PROCESSED_DIR / "compound_initial_offsets_2024.csv", index=False)
        loro_metrics.to_csv(PROCESSED_DIR / "loro_cv_fold_metrics_2024.csv", index=False)
        oof_df.to_csv(PROCESSED_DIR / "loro_oof_predictions_2024.csv", index=False)
        error_dict["combined_regimes"].to_csv(
            PROCESSED_DIR / "error_analysis_by_regime_2024.csv", index=False
        )
        error_dict["worst_stints"].to_csv(
            PROCESSED_DIR / "error_analysis_worst_stints_2024.csv", index=False
        )

    return {
        "baseline_table": baseline_table,
        "compound_offsets": compound_offsets,
        "loro_metrics": loro_metrics,
        "oof_predictions": oof_df,
        "error_regimes": error_dict["combined_regimes"],
        "worst_stints": error_dict["worst_stints"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_stage3_pipeline(save_outputs=True)
    print("\n=== BASELINE LINEAR REGRESSION PER CIRCUIT & COMPOUND ===")
    print(res["baseline_table"].to_string(index=False))
    print("\n=== 5-FOLD LEAVE-ONE-RACE-OUT CV METRICS (SECONDS) ===")
    print(res["loro_metrics"].to_string(index=False))
    print("\n=== ERROR ANALYSIS BY REGIME (SECONDS) ===")
    print(res["error_regimes"].to_string(index=False))
    print("\n=== TOP 10 HIGHEST-ERROR STINTS ===")
    print(res["worst_stints"].to_string(index=False))
