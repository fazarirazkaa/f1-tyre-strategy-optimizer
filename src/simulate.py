"""Stage 4: Formula 1 Pit Stop Strategy Simulator & Sensitivity Analyzer.

This module implements:
1. Validation of FIA Sporting Regulations Article 30.5(m) for dry races (requiring at
   least two distinct dry tyre compounds per race: e.g., SOFT + MEDIUM, MEDIUM + HARD).
2. Fast, physics-informed lap-by-lap race time simulation (`CircuitStrategySimulator`)
   combining:
   - Empirical circuit baseline pace and calibrated fuel burn penalty (`gamma_c * (N - L)`)
   - Calibrated fresh-tyre compound pace offsets (`SOFT < MEDIUM < HARD` at `TyreLife = 1`)
   - Circuit-calibrated + non-linear `HybridPhysicsBoostingModel` wear curves
   - Progressive thermal/carcass cliff penalty when a stint exceeds sustainable compound
     wear limits (`SOFT > 18`, `MEDIUM > 28`, `HARD > 40` laps)
   - Empirical circuit-specific green-flag pit stop loss (`median([in_lap + out_lap] - 2 * normal_lap)`)
3. Exhaustive optimal pit window search for 1-stop and 2-stop strategies.
4. Two-way sensitivity analysis over Pit Stop Loss (`+/- 5s`) and Degradation Rate
   Multiplier (`0.7x` to `1.5x`) to identify strategic tipping points (1-stop vs. 2-stop).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.load import DRY_COMPOUNDS, PROCESSED_DIR
from src.model import (
    REFERENCE_LAP_SECONDS,
    HybridPhysicsBoostingModel,
    fit_full_models,
    prepare_modeling_features,
)

logger = logging.getLogger(__name__)

# Minimum laps required per stint in optimal strategy search
MIN_STINT_LAPS: int = 6

# Standing start penalty on Lap 1 (identical across strategies, included for realistic total race time)
STANDING_START_PENALTY_S: float = 4.5

# Empirical tyre wear cliff thresholds (in 85s-equivalent scaled laps) and quadratic cliff coefficients
# Derived from 90th-percentile sustainable stint lengths in 2024 dry races:
# Beyond these thresholds, thermal degradation and carcass wear accelerate non-linearly.
CLIFF_ONSET_SCALED_LAPS: Dict[str, float] = {
    "SOFT": 14.0,
    "MEDIUM": 23.0,
    "HARD": 42.0,
}
CLIFF_QUADRATIC_COEF: Dict[str, float] = {
    "SOFT": 0.0150,
    "MEDIUM": 0.0095,
    "HARD": 0.0045,
}
SECOND_SAME_COMPOUND_Q_AGE_LAPS: Dict[str, int] = {
    "SOFT": 3,    # Second Soft set comes from scrubbed Q1/Q2/Q3 allocation (+3 laps)
    "MEDIUM": 2,  # Second Medium set typically scrubbed in FP/Sprint (+2 laps)
    "HARD": 0,    # Teams often save two fresh Hard sets on high-wear tracks (e.g., Suzuka/Bahrain)
}


@dataclass(frozen=True)
class StrategySpec:
    """Specification of a 1-stop or 2-stop dry race strategy."""

    circuit: str
    compounds: Tuple[str, ...]
    pit_laps: Tuple[int, ...]

    @property
    def num_stops(self) -> int:
        return len(self.pit_laps)

    @property
    def strategy_type(self) -> str:
        return f"{self.num_stops}-Stop"

    @property
    def label(self) -> str:
        comp_str = " → ".join(self.compounds)
        pits_str = ", ".join(f"L{p}" for p in self.pit_laps)
        return f"{self.num_stops}-Stop ({comp_str} | Pit {pits_str})"

    @property
    def short_compound_code(self) -> str:
        return "-".join(c[0] for c in self.compounds)


def validate_dry_strategy(
    compounds: Sequence[str],
    pit_laps: Sequence[int],
    total_race_laps: int,
    min_stint_laps: int = MIN_STINT_LAPS,
) -> Tuple[bool, str]:
    """Verify FIA dry tyre rule (at least 2 distinct compounds) and valid pit lap boundaries."""
    if len(compounds) != len(pit_laps) + 1:
        return False, "Number of stints (compounds) must equal number of pit stops + 1."

    if len(compounds) < 2:
        return False, "Strategy must include at least 1 pit stop (minimum 2 stints)."

    for c in compounds:
        if c not in DRY_COMPOUNDS:
            return False, f"Invalid dry compound '{c}'. Allowed: {DRY_COMPOUNDS}."

    # FIA Sporting Regulations Article 30.5(m): must use at least 2 different dry specifications
    if len(set(compounds)) < 2:
        return (
            False,
            f"Illegal FIA dry strategy ({' → '.join(compounds)}): must use at least two different compounds.",
        )

    boundaries = [0] + list(pit_laps) + [total_race_laps]
    for i in range(len(boundaries) - 1):
        stint_len = boundaries[i + 1] - boundaries[i]
        if stint_len < min_stint_laps:
            return (
                False,
                f"Stint {i + 1} length ({stint_len} laps) is shorter than minimum {min_stint_laps} laps.",
            )

    return True, "Valid FIA dry strategy."


def _format_race_time(total_seconds: float) -> str:
    """Format total race seconds as 'H:MM:SS.ss'."""
    hours = int(total_seconds // 3600)
    rem = total_seconds - hours * 3600
    minutes = int(rem // 60)
    secs = rem - minutes * 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


class CircuitStrategySimulator:
    """Precomputed, vectorized strategy simulator for a specific Grand Prix circuit."""

    def __init__(
        self,
        circuit: str,
        clean_df: pd.DataFrame,
        pit_summary_df: pd.DataFrame,
        fuel_calib_df: pd.DataFrame,
        baseline_table: pd.DataFrame,
        compound_offsets_df: pd.DataFrame,
        hybrid_model: HybridPhysicsBoostingModel,
    ) -> None:
        self.circuit = circuit
        c_df = clean_df[clean_df["Circuit"] == circuit]
        if c_df.empty:
            raise ValueError(f"No data found for circuit '{circuit}'.")

        self.event_name: str = str(c_df["EventName"].iloc[0])
        self.total_laps: int = int(c_df["TotalRaceLaps"].iloc[0])
        self.circuit_base_pace: float = float(c_df["CircuitBaselinePace"].iloc[0])
        self.lap_scale: float = self.circuit_base_pace / REFERENCE_LAP_SECONDS
        self.track_temp: float = float(c_df["TrackTemp"].median())
        self.air_temp: float = float(c_df["AirTemp"].median())
        self.temp_delta: float = self.track_temp - self.air_temp

        pit_row = pit_summary_df[pit_summary_df["Circuit"] == circuit].iloc[0]
        self.empirical_pit_loss: float = float(pit_row["MedianPitLossSeconds"])
        self.pit_loss_iqr: float = float(pit_row["IQRPitLossSeconds"])

        fuel_row = fuel_calib_df[fuel_calib_df["Circuit"] == circuit].iloc[0]
        self.fuel_burn_kg_per_lap: float = float(fuel_row["FuelBurnKgPerLap"])
        self.fuel_penalty_s_per_lap: float = float(fuel_row["FuelPenaltySecondsPerLap"])

        # Compound fresh offsets and circuit-specific slopes
        self.fresh_offsets: Dict[str, float] = {}
        self.circuit_slopes: Dict[str, float] = {}
        for comp in DRY_COMPOUNDS:
            off_row = compound_offsets_df[
                (compound_offsets_df["Circuit"] == circuit)
                & (compound_offsets_df["Compound"] == comp)
            ]
            self.fresh_offsets[comp] = (
                float(off_row["FreshOffsetVsMediumSeconds"].iloc[0])
                if not off_row.empty
                else 0.0
            )
            slp_row = baseline_table[
                (baseline_table["Circuit"] == circuit)
                & (baseline_table["Compound"] == comp)
            ]
            self.circuit_slopes[comp] = (
                float(slp_row["Regularized_Slope_s_per_lap"].iloc[0])
                if not slp_row.empty
                else 0.06
            )

        # Enforce physical C-compound wear hierarchy within the pre-cliff domain:
        # Corrects for right-censoring where HARD is run 35+ laps while MEDIUM is stopped at ~18 laps
        med_s = max(self.circuit_slopes["MEDIUM"], 0.88 * self.circuit_slopes["HARD"] + 0.012)
        hard_s = min(self.circuit_slopes["HARD"], 0.80 * med_s)
        soft_s = max(self.circuit_slopes["SOFT"], 1.22 * med_s)
        self.circuit_slopes["HARD"] = round(hard_s, 4)
        self.circuit_slopes["MEDIUM"] = round(med_s, 4)
        self.circuit_slopes["SOFT"] = round(soft_s, 4)

        # Precompute deterministic base + fuel lap time per race lap L (1..N)
        laps_arr = np.arange(1, self.total_laps + 1, dtype=float)
        rem_laps = self.total_laps - laps_arr
        self.base_plus_fuel_by_lap: np.ndarray = (
            self.circuit_base_pace + rem_laps * self.fuel_penalty_s_per_lap
        )
        self.base_plus_fuel_by_lap[0] += STANDING_START_PENALTY_S

        # Precompute 3D degradation wear table: wear_lookup[comp][lap_idx, tyre_life_idx]
        # where lap_idx in 0..N-1 (LapNumber = lap_idx + 1) and tyre_life_idx in 0..N-1 (TyreLife = tyre_life_idx + 1)
        self.wear_lookup: Dict[str, np.ndarray] = self._build_wear_lookup(hybrid_model)

    def _build_wear_lookup(
        self, hybrid_model: HybridPhysicsBoostingModel
    ) -> Dict[str, np.ndarray]:
        """Precompute fuel-adjusted tyre degradation delta for every (Compound, LapNumber, TyreLife)."""
        n = self.total_laps
        comp_code_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
        lookup: Dict[str, np.ndarray] = {}

        for comp in DRY_COMPOUNDS:
            grid_lap, grid_tl = np.meshgrid(
                np.arange(1, n + 1, dtype=int),
                np.arange(1, n + 1, dtype=int),
                indexing="ij",
            )
            flat_lap = grid_lap.ravel()
            flat_tl = grid_tl.ravel()
            flat_stint_age = flat_tl - 1

            rem_laps = n - flat_lap
            est_fuel_kg = rem_laps * self.fuel_burn_kg_per_lap

            sim_df = pd.DataFrame(
                {
                    "Circuit": self.circuit,
                    "Driver": "SIM",
                    "Stint": 1,
                    "Compound": comp,
                    "CompoundCode": comp_code_map[comp],
                    "FreshTyre": True,
                    "LapNumber": flat_lap,
                    "TotalRaceLaps": n,
                    "RaceProgress": flat_lap / float(n),
                    "TyreLife": flat_tl,
                    "StintMinTyreLife": 1,
                    "StintLapAge": flat_stint_age,
                    "CircuitBaselinePace": self.circuit_base_pace,
                    "TrackTemp": self.track_temp,
                    "AirTemp": self.air_temp,
                    "EstimatedFuelKg": est_fuel_kg,
                }
            )
            feat_sim = prepare_modeling_features(sim_df)
            hybrid_raw = hybrid_model.predict(feat_sim).reshape(n, n)

            # Anchor hybrid curve so TyreLife=1 has zero cumulative wear (pure wear relative to fresh lap 1)
            hybrid_anchored = np.maximum(0.0, hybrid_raw - hybrid_raw[:, [0]])

            # Circuit-calibrated linear wear term: slope * (TyreLife - 1)
            tl_zero = np.arange(0, n, dtype=float)[None, :]
            circuit_linear = self.circuit_slopes[comp] * tl_zero

            # Progressive quadratic cliff penalty beyond sustainable compound life (Decision A2)
            scaled_tl = (tl_zero + 1.0) * self.lap_scale
            cliff_excess = np.maximum(0.0, scaled_tl - CLIFF_ONSET_SCALED_LAPS[comp])
            cliff_penalty = CLIFF_QUADRATIC_COEF[comp] * (cliff_excess ** 2)

            # Combine 50% circuit-specific slope + 50% non-linear Hybrid model + cliff penalty
            wear_mat = 0.55 * circuit_linear + 0.45 * hybrid_anchored + cliff_penalty
            lookup[comp] = wear_mat

        return lookup

    def simulate_strategy(
        self,
        compounds: Sequence[str],
        pit_laps: Sequence[int],
        pit_loss_override: Optional[float] = None,
        degradation_multiplier: float = 1.0,
        return_lap_trace: bool = False,
    ) -> Dict[str, object]:
        """Simulate a single 1-stop or 2-stop strategy and return total time & optionally lap trace."""
        is_valid, msg = validate_dry_strategy(
            compounds, pit_laps, self.total_laps, min_stint_laps=2
        )
        if not is_valid:
            raise ValueError(msg)

        pit_loss = (
            float(pit_loss_override)
            if pit_loss_override is not None
            else self.empirical_pit_loss
        )
        boundaries = [0] + list(pit_laps) + [self.total_laps]
        n_stops = len(pit_laps)

        total_driving_seconds = 0.0
        stint_lengths: List[int] = []
        lap_records: List[Dict[str, object]] = []

        seen_counts: Dict[str, int] = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
        for stint_idx, comp in enumerate(compounds):
            start_lap = boundaries[stint_idx] + 1
            end_lap = boundaries[stint_idx + 1]
            stint_len = end_lap - start_lap + 1
            stint_lengths.append(stint_len)

            # A second set of SOFT or MEDIUM used in the race carries prior scrub age
            q_offset = (
                SECOND_SAME_COMPOUND_Q_AGE_LAPS.get(comp, 0)
                if seen_counts.get(comp, 0) >= 1
                else 0
            )
            seen_counts[comp] = seen_counts.get(comp, 0) + 1

            lap_indices = np.arange(start_lap - 1, end_lap, dtype=int)
            tl_indices = np.clip(np.arange(q_offset, stint_len + q_offset, dtype=int), 0, self.total_laps - 1)

            base_fuel_vec = self.base_plus_fuel_by_lap[lap_indices]
            fresh_off = self.fresh_offsets[comp]
            wear_vec = (
                degradation_multiplier
                * self.wear_lookup[comp][lap_indices, tl_indices]
            )
            lap_times_vec = base_fuel_vec + fresh_off + wear_vec
            total_driving_seconds += float(np.sum(lap_times_vec))

            if return_lap_trace:
                for k in range(stint_len):
                    ln = int(start_lap + k)
                    is_pit_in = ln in pit_laps
                    lap_records.append(
                        {
                            "Circuit": self.circuit,
                            "LapNumber": ln,
                            "Stint": stint_idx + 1,
                            "Compound": comp,
                            "TyreLife": k + 1,
                            "PredictedDrivingLapSeconds": round(float(lap_times_vec[k]), 3),
                            "PredictedDegradationDeltaSeconds": round(
                                float(fresh_off + wear_vec[k]), 3
                            ),
                            "FuelPenaltySeconds": round(
                                float(
                                    (self.total_laps - ln) * self.fuel_penalty_s_per_lap
                                ),
                                3,
                            ),
                            "PitLossAppliedSeconds": round(pit_loss, 2) if is_pit_in else 0.0,
                            "EffectiveLapSeconds": round(
                                float(lap_times_vec[k] + (pit_loss if is_pit_in else 0.0)),
                                3,
                            ),
                        }
                    )

        total_pit_loss_seconds = n_stops * pit_loss
        total_race_seconds = total_driving_seconds + total_pit_loss_seconds
        spec = StrategySpec(
            circuit=self.circuit,
            compounds=tuple(compounds),
            pit_laps=tuple(int(p) for p in pit_laps),
        )

        result: Dict[str, object] = {
            "Circuit": self.circuit,
            "EventName": self.event_name,
            "StrategyType": spec.strategy_type,
            "NumStops": n_stops,
            "CompoundSequence": " → ".join(compounds),
            "ShortCode": spec.short_compound_code,
            "StartingCompound": compounds[0],
            "PitLaps": ", ".join(str(p) for p in pit_laps),
            "PitLap1": int(pit_laps[0]),
            "PitLap2": int(pit_laps[1]) if n_stops >= 2 else None,
            "StintLengths": "-".join(str(s) for s in stint_lengths),
            "TotalDrivingSeconds": round(total_driving_seconds, 2),
            "TotalPitLossSeconds": round(total_pit_loss_seconds, 2),
            "TotalRaceSeconds": round(total_race_seconds, 2),
            "FormattedRaceTime": _format_race_time(total_race_seconds),
            "PitLossUsedSeconds": round(pit_loss, 2),
            "DegradationMultiplier": round(degradation_multiplier, 2),
            "StrategyLabel": spec.label,
        }

        if return_lap_trace:
            trace_df = pd.DataFrame(lap_records)
            trace_df["CumulativeRaceSeconds"] = trace_df["EffectiveLapSeconds"].cumsum().round(3)
            result["LapTrace"] = trace_df

        return result

    def _build_stint_cost_matrices(
        self, degradation_multiplier: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Precompute exact driving time (s) for any stint `[start_lap, end_lap]` (0-indexed boundaries).

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            - `cost_fresh[comp][b_start, b_end]`: driving time for a fresh set from lap `b_start+1` to `b_end`
            - `cost_used[comp][b_start, b_end]`: driving time for a 2nd set of `comp` with scrub offset
        """
        n = self.total_laps
        cost_fresh: Dict[str, np.ndarray] = {
            c: np.full((n + 1, n + 1), np.inf, dtype=float) for c in DRY_COMPOUNDS
        }
        cost_used: Dict[str, np.ndarray] = {
            c: np.full((n + 1, n + 1), np.inf, dtype=float) for c in DRY_COMPOUNDS
        }

        for b_start in range(0, n):
            max_len = n - b_start
            lap_idx = np.arange(b_start, n, dtype=int)
            tl_idx = np.arange(0, max_len, dtype=int)
            base_vec = self.base_plus_fuel_by_lap[lap_idx]

            for comp in DRY_COMPOUNDS:
                fresh_off = self.fresh_offsets[comp]
                wear_vec = (
                    degradation_multiplier * self.wear_lookup[comp][lap_idx, tl_idx]
                )
                lap_times = base_vec + fresh_off + wear_vec
                cost_fresh[comp][b_start, b_start + 1 : n + 1] = np.cumsum(lap_times)

                q_off = SECOND_SAME_COMPOUND_Q_AGE_LAPS.get(comp, 0)
                tl_used_idx = np.clip(tl_idx + q_off, 0, n - 1)
                wear_used = (
                    degradation_multiplier * self.wear_lookup[comp][lap_idx, tl_used_idx]
                )
                cost_used[comp][b_start, b_start + 1 : n + 1] = np.cumsum(
                    base_vec + fresh_off + wear_used
                )

        return cost_fresh, cost_used

    def search_optimal_strategies(
        self,
        pit_loss_override: Optional[float] = None,
        degradation_multiplier: float = 1.0,
        min_stint_laps: int = MIN_STINT_LAPS,
        starting_compound_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Exhaustively evaluate all valid 1-stop and 2-stop FIA dry strategies at exact 1-lap resolution."""
        n = self.total_laps
        pit_loss = (
            float(pit_loss_override)
            if pit_loss_override is not None
            else self.empirical_pit_loss
        )
        cost_fresh, cost_used = self._build_stint_cost_matrices(degradation_multiplier)
        candidates: List[Dict[str, object]] = []

        # 1. All legal 1-stop compound pairs (6 ordered pairs with 2 distinct compounds)
        one_stop_pairs = [
            (c1, c2)
            for c1, c2 in itertools.product(DRY_COMPOUNDS, repeat=2)
            if c1 != c2
        ]
        p1_range_1stop = np.arange(min_stint_laps, n - min_stint_laps + 1, dtype=int)
        for c1, c2 in one_stop_pairs:
            if starting_compound_filter and c1 != starting_compound_filter:
                continue
            for p1 in p1_range_1stop:
                t_drive = float(cost_fresh[c1][0, p1] + cost_fresh[c2][p1, n])
                t_total = t_drive + pit_loss
                spec = StrategySpec(self.circuit, (c1, c2), (int(p1),))
                candidates.append(
                    {
                        "Circuit": self.circuit,
                        "EventName": self.event_name,
                        "StrategyType": "1-Stop",
                        "NumStops": 1,
                        "CompoundSequence": f"{c1} → {c2}",
                        "ShortCode": spec.short_compound_code,
                        "StartingCompound": c1,
                        "PitLaps": str(int(p1)),
                        "PitLap1": int(p1),
                        "PitLap2": None,
                        "StintLengths": f"{p1}-{n - p1}",
                        "TotalDrivingSeconds": round(t_drive, 2),
                        "TotalPitLossSeconds": round(pit_loss, 2),
                        "TotalRaceSeconds": round(t_total, 2),
                        "FormattedRaceTime": _format_race_time(t_total),
                        "PitLossUsedSeconds": round(pit_loss, 2),
                        "DegradationMultiplier": round(degradation_multiplier, 2),
                        "StrategyLabel": spec.label,
                    }
                )

        # 2. All legal 2-stop compound triplets (24 triplets with >= 2 distinct compounds)
        two_stop_triplets = [
            (c1, c2, c3)
            for c1, c2, c3 in itertools.product(DRY_COMPOUNDS, repeat=3)
            if len({c1, c2, c3}) >= 2
        ]
        for c1, c2, c3 in two_stop_triplets:
            if starting_compound_filter and c1 != starting_compound_filter:
                continue
            mat1 = cost_fresh[c1]
            mat2 = cost_used[c2] if (c2 == c1) else cost_fresh[c2]
            mat3 = cost_used[c3] if (c3 in (c1, c2)) else cost_fresh[c3]

            # Vectorized exact search over all valid (p1, p2) pairs for this compound triplet
            best_for_triplet: List[Tuple[float, int, int]] = []
            for p1 in range(min_stint_laps, n - 2 * min_stint_laps + 1):
                p2_arr = np.arange(p1 + min_stint_laps, n - min_stint_laps + 1, dtype=int)
                if p2_arr.size == 0:
                    continue
                t_drives = mat1[0, p1] + mat2[p1, p2_arr] + mat3[p2_arr, n]
                min_idx = int(np.argmin(t_drives))
                best_for_triplet.append((float(t_drives[min_idx]), int(p1), int(p2_arr[min_idx])))

            best_for_triplet.sort(key=lambda x: x[0])
            # Record the top 3 pit window variations for each 2-stop compound sequence
            for t_drive, p1_opt, p2_opt in best_for_triplet[:3]:
                t_total = t_drive + 2.0 * pit_loss
                spec = StrategySpec(self.circuit, (c1, c2, c3), (p1_opt, p2_opt))
                candidates.append(
                    {
                        "Circuit": self.circuit,
                        "EventName": self.event_name,
                        "StrategyType": "2-Stop",
                        "NumStops": 2,
                        "CompoundSequence": f"{c1} → {c2} → {c3}",
                        "ShortCode": spec.short_compound_code,
                        "StartingCompound": c1,
                        "PitLaps": f"{p1_opt}, {p2_opt}",
                        "PitLap1": p1_opt,
                        "PitLap2": p2_opt,
                        "StintLengths": f"{p1_opt}-{p2_opt - p1_opt}-{n - p2_opt}",
                        "TotalDrivingSeconds": round(t_drive, 2),
                        "TotalPitLossSeconds": round(2.0 * pit_loss, 2),
                        "TotalRaceSeconds": round(t_total, 2),
                        "FormattedRaceTime": _format_race_time(t_total),
                        "PitLossUsedSeconds": round(pit_loss, 2),
                        "DegradationMultiplier": round(degradation_multiplier, 2),
                        "StrategyLabel": spec.label,
                    }
                )

        df_all = (
            pd.DataFrame(candidates)
            .sort_values("TotalRaceSeconds")
            .reset_index(drop=True)
        )
        min_time = float(df_all["TotalRaceSeconds"].iloc[0])
        df_all["DeltaToOptimalSeconds"] = (df_all["TotalRaceSeconds"] - min_time).round(2)
        df_all["Rank"] = np.arange(1, len(df_all) + 1)
        return df_all

    def run_sensitivity_grid(
        self,
        pit_loss_deltas: Sequence[float] = (-4.0, -2.0, 0.0, +2.0, +4.0),
        deg_multipliers: Sequence[float] = (0.75, 0.90, 1.00, 1.15, 1.30),
    ) -> pd.DataFrame:
        """Evaluate 1-stop vs. 2-stop trade-off across a 2D grid of Pit Loss & Degradation Rate."""
        rows: List[Dict[str, object]] = []
        for d_pit in pit_loss_deltas:
            test_pit_loss = max(12.0, self.empirical_pit_loss + float(d_pit))
            for deg_mult in deg_multipliers:
                ranked = self.search_optimal_strategies(
                    pit_loss_override=test_pit_loss,
                    degradation_multiplier=float(deg_mult),
                )
                best_1stop = ranked[ranked["NumStops"] == 1].iloc[0]
                best_2stop = ranked[ranked["NumStops"] == 2].iloc[0]
                overall_best = ranked.iloc[0]

                time_1 = float(best_1stop["TotalRaceSeconds"])
                time_2 = float(best_2stop["TotalRaceSeconds"])
                # Positive gap means 2-stop is faster by (time_1 - time_2) seconds
                two_stop_advantage_s = round(time_1 - time_2, 2)

                rows.append(
                    {
                        "Circuit": self.circuit,
                        "PitLossDeltaSeconds": float(d_pit),
                        "PitLossSeconds": round(test_pit_loss, 2),
                        "DegradationMultiplier": float(deg_mult),
                        "OptimalStrategyType": str(overall_best["StrategyType"]),
                        "OptimalCompoundSequence": str(overall_best["CompoundSequence"]),
                        "OptimalPitLaps": str(overall_best["PitLaps"]),
                        "Best1StopTimeSeconds": round(time_1, 2),
                        "Best1StopCompounds": str(best_1stop["CompoundSequence"]),
                        "Best1StopPitLaps": str(best_1stop["PitLaps"]),
                        "Best2StopTimeSeconds": round(time_2, 2),
                        "Best2StopCompounds": str(best_2stop["CompoundSequence"]),
                        "Best2StopPitLaps": str(best_2stop["PitLaps"]),
                        "TwoStopAdvantageSeconds": two_stop_advantage_s,
                    }
                )
        return pd.DataFrame(rows)


def build_all_circuit_simulators(
    clean_df: Optional[pd.DataFrame] = None,
) -> Dict[str, CircuitStrategySimulator]:
    """Instantiate `CircuitStrategySimulator` for all 5 active circuits."""
    if clean_df is None:
        clean_df = pd.read_csv(PROCESSED_DIR / "clean_laps_2024.csv")
    pit_summary_df = pd.read_csv(PROCESSED_DIR / "pit_loss_summary_2024.csv")
    fuel_calib_df = pd.read_csv(PROCESSED_DIR / "fuel_calibration_2024.csv")

    models_dict = fit_full_models(clean_df)
    baseline_table = models_dict["baseline_table"]
    compound_offsets_df = models_dict["compound_offsets"]
    hybrid_model = models_dict["hybrid"]

    simulators: Dict[str, CircuitStrategySimulator] = {}
    for circuit in clean_df["Circuit"].unique():
        simulators[circuit] = CircuitStrategySimulator(
            circuit=circuit,
            clean_df=clean_df,
            pit_summary_df=pit_summary_df,
            fuel_calib_df=fuel_calib_df,
            baseline_table=baseline_table,
            compound_offsets_df=compound_offsets_df,
            hybrid_model=hybrid_model,
        )
    return simulators


def run_stage4_pipeline(save_outputs: bool = True) -> Dict[str, pd.DataFrame]:
    """Run optimal strategy search and sensitivity analysis across all 5 circuits."""
    simulators = build_all_circuit_simulators()

    summary_rows: List[Dict[str, object]] = []
    top_strategies_frames: List[pd.DataFrame] = []
    sensitivity_frames: List[pd.DataFrame] = []

    for circuit, sim in simulators.items():
        logger.info("Optimizing 1-stop and 2-stop strategies for %s...", circuit)
        ranked = sim.search_optimal_strategies()

        # Keep top 5 1-stop and top 5 2-stop per circuit for dashboard & comparison tables
        top_1stop = ranked[ranked["NumStops"] == 1].head(5)
        top_2stop = ranked[ranked["NumStops"] == 2].head(5)
        top_combined = (
            pd.concat([top_1stop, top_2stop], ignore_index=True)
            .sort_values("TotalRaceSeconds")
            .reset_index(drop=True)
        )
        top_strategies_frames.append(top_combined)

        best_overall = ranked.iloc[0]
        best_1 = top_1stop.iloc[0]
        best_2 = top_2stop.iloc[0]

        summary_rows.append(
            {
                "Circuit": circuit,
                "EventName": sim.event_name,
                "TotalRaceLaps": sim.total_laps,
                "EmpiricalPitLossSeconds": sim.empirical_pit_loss,
                "RecommendedStrategy": str(best_overall["StrategyType"]),
                "OptimalCompounds": str(best_overall["CompoundSequence"]),
                "OptimalPitLaps": str(best_overall["PitLaps"]),
                "OptimalStintLengths": str(best_overall["StintLengths"]),
                "OptimalTotalTime": str(best_overall["FormattedRaceTime"]),
                "Best1StopCompounds": str(best_1["CompoundSequence"]),
                "Best1StopPitLap": int(best_1["PitLap1"]),
                "Best1StopTime": str(best_1["FormattedRaceTime"]),
                "Best2StopCompounds": str(best_2["CompoundSequence"]),
                "Best2StopPitLaps": str(best_2["PitLaps"]),
                "Best2StopTime": str(best_2["FormattedRaceTime"]),
                "TimeGap_2Stop_vs_1Stop_s": round(
                    float(best_2["TotalRaceSeconds"]) - float(best_1["TotalRaceSeconds"]),
                    2,
                ),
            }
        )

        sens_df = sim.run_sensitivity_grid()
        sensitivity_frames.append(sens_df)

    optimal_summary_df = pd.DataFrame(summary_rows)
    top_strategies_df = pd.concat(top_strategies_frames, ignore_index=True)
    sensitivity_df = pd.concat(sensitivity_frames, ignore_index=True)

    if save_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        optimal_summary_df.to_csv(PROCESSED_DIR / "strategy_optimal_summary_2024.csv", index=False)
        top_strategies_df.to_csv(PROCESSED_DIR / "strategy_top_candidates_2024.csv", index=False)
        sensitivity_df.to_csv(PROCESSED_DIR / "strategy_sensitivity_grid_2024.csv", index=False)

    return {
        "optimal_summary": optimal_summary_df,
        "top_strategies": top_strategies_df,
        "sensitivity": sensitivity_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_stage4_pipeline(save_outputs=True)
    print("\n=== OPTIMAL PIT STRATEGY SUMMARY PER CIRCUIT ===")
    print(res["optimal_summary"].to_string(index=False))
    print("\n=== SENSITIVITY ANALYSIS SAMPLE (DEGRADATION MULTIPLIER = 1.0 vs 1.3) ===")
    sens = res["sensitivity"]
    print(
        sens[sens["PitLossDeltaSeconds"].isin([-2.0, 0.0, 2.0])][
            [
                "Circuit",
                "PitLossSeconds",
                "DegradationMultiplier",
                "OptimalStrategyType",
                "OptimalCompoundSequence",
                "OptimalPitLaps",
                "TwoStopAdvantageSeconds",
            ]
        ].to_string(index=False)
    )
