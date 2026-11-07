"""Data loading, cleaning, and empirical pit stop loss estimation using FastF1.

This module implements Stage 2 data acquisition and filtering:
1. Configures FastF1 local cache at `data/cache/`.
2. Loads 2024 Race ('R') sessions for the five target dry Grand Prix events.
   Uses `fastf1.get_session()` and `fastf1.Cache.requests_get()` with automatic
   fallback to the public FIA LiveTiming mirror (`api.openf1.org/v1/`) when
   `livetiming.formula1.com` requires a paid F1TV subscription token (FastF1 >= 3.9).
3. Filters non-representative laps:
   - Lap 1 (standing start & opening-lap traffic)
   - In-laps (`PitInTime` is not null) and Out-laps (`PitOutTime` is not null)
   - Track interruptions: Safety Car ('4'), Red Flag ('5'), VSC ('6', '7')
   - FastF1 `IsAccurate == False` or missing `LapTime` / `TyreLife`
   - Stint-aware outlier filtering (Decision A1: 107% of driver fastest clean lap
     combined with per-stint linear degradation residual IQR filtering)
4. Estimates empirical green-flag pit stop loss per circuit from data:
   `median( [in_lap + out_lap] - 2 * driver_normal_lap_time )`
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fastf1
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reproducibility constant
RANDOM_SEED: int = 42

# Default paths relative to project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Five initial dry races specified in the project prompt (kept for EDA audit reference)
INITIAL_RACES_2024: Dict[str, str] = {
    "Bahrain": "Bahrain Grand Prix",
    "Spain": "Spanish Grand Prix",
    "Hungary": "Hungarian Grand Prix",
    "Monza": "Italian Grand Prix",
    "Abu Dhabi": "Abu Dhabi Grand Prix",
}

# Active 5 dry races after EDA verification:
# Bahrain (0 Medium stints) and Monza (only 1 clean Soft lap) were replaced with
# Japan (Suzuka) and Netherlands (Zandvoort) so all 5 races have full 3-compound coverage.
DEFAULT_RACES_2024: Dict[str, str] = {
    "Japan": "Japanese Grand Prix",
    "Spain": "Spanish Grand Prix",
    "Hungary": "Hungarian Grand Prix",
    "Netherlands": "Dutch Grand Prix",
    "Abu Dhabi": "Abu Dhabi Grand Prix",
}

# Mapping from circuit key / EventName to OpenF1 country/location identifiers for 2024
OPENF1_LOCATION_MAP_2024: Dict[str, Tuple[str, str]] = {
    "Bahrain": ("Bahrain", "Sakhir"),
    "Saudi Arabia": ("Saudi Arabia", "Jeddah"),
    "Australia": ("Australia", "Melbourne"),
    "Japan": ("Japan", "Suzuka"),
    "China": ("China", "Shanghai"),
    "Miami": ("United States", "Miami"),
    "Imola": ("Italy", "Imola"),
    "Monaco": ("Monaco", "Monaco"),
    "Canada": ("Canada", "Montréal"),
    "Spain": ("Spain", "Barcelona"),
    "Austria": ("Austria", "Spielberg"),
    "Silverstone": ("United Kingdom", "Silverstone"),
    "Hungary": ("Hungary", "Budapest"),
    "Belgium": ("Belgium", "Spa-Francorchamps"),
    "Netherlands": ("Netherlands", "Zandvoort"),
    "Monza": ("Italy", "Monza"),
    "Azerbaijan": ("Azerbaijan", "Baku"),
    "Singapore": ("Singapore", "Marina Bay"),
    "Austin": ("United States", "Austin"),
    "Mexico": ("Mexico", "Mexico City"),
    "Brazil": ("Brazil", "São Paulo"),
    "Las Vegas": ("United States", "Las Vegas"),
    "Qatar": ("Qatar", "Lusail"),
    "Abu Dhabi": ("United Arab Emirates", "Yas Island"),
}

OPENF1_BASE_URL: str = "https://api.openf1.org/v1"

# Standard dry tyre compounds in Formula 1
DRY_COMPOUNDS: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

# FIA TrackStatus codes indicating neutralized/interrupted track conditions:
# '1' = AllClear, '2' = Yellow, '4' = SafetyCar, '5' = RedFlag, '6' = VSC, '7' = VSCEnding
INTERRUPTED_STATUS_CODES: Tuple[str, ...] = ("4", "5", "6", "7")

# Filtering thresholds (documented explicitly for transparency)
PACE_THRESHOLD_RATIO: float = 1.07  # 107% rule relative to driver's fastest clean lap
STINT_RESIDUAL_IQR_MULTIPLIER: float = 1.5  # Multiplier for residual IQR within a stint
MIN_STINT_LAPS_FOR_TREND: int = 4  # Minimum laps in a stint to fit local linear trend
PIT_LOSS_MIN_SECONDS: float = 14.0  # Lower bound for valid green-flag pit loss (s)
PIT_LOSS_MAX_SECONDS: float = 38.0  # Upper bound to exclude wing changes / penalties (s)


def setup_fastf1_cache(cache_dir: Optional[Path] = None) -> Path:
    """Enable FastF1 cache directory, creating it if necessary."""
    target_dir = cache_dir or CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):
        fastf1.Cache.configure(cache_dir=str(target_dir))
    else:
        fastf1.Cache.enable_cache(str(target_dir))
    return target_dir


def _has_interrupted_track_status(status_val: object) -> bool:
    """Return True if TrackStatus string contains SC ('4'), Red ('5'), or VSC ('6', '7')."""
    if pd.isna(status_val):
        return False
    status_str = str(status_val)
    return any(code in status_str for code in INTERRUPTED_STATUS_CODES)


def _cached_json_get(url: str, max_retries: int = 6) -> list | dict:
    """Fetch JSON through FastF1's SQLite HTTP cache (`data/cache/fastf1_http_cache.sqlite`)."""
    import requests as _requests

    for attempt in range(max_retries):
        resp = fastf1.Cache.requests_get(url, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "detail" not in data:
                return data
        # If a 429 rate-limit response was cached, wait and fetch fresh
        time.sleep(1.5 * (attempt + 1))
        fresh_resp = _requests.get(url, timeout=25)
        if fresh_resp.status_code == 200:
            data = fresh_resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "detail" not in data:
                return data
    raise RuntimeError(f"Failed to fetch data from {url} (HTTP {resp.status_code})")


def _build_neutralized_laps_from_race_control(rc_messages: list) -> Tuple[Dict[int, str], Set[Tuple[int, int]]]:
    """Derive FIA TrackStatus code per lap and deleted laps from FIA Race Control stream.

    FIA TrackStatus codes:
    - '1': Track Clear
    - '2': Yellow / Double Yellow Flag
    - '4': Safety Car deployed
    - '5': Red Flag
    - '6': Virtual Safety Car (VSC) deployed
    - '7': VSC ending
    """
    lap_status: Dict[int, Set[str]] = {}
    deleted_driver_laps: Set[Tuple[int, int]] = set()

    sc_active = False
    vsc_active = False
    current_max_lap = 1

    for msg in rc_messages:
        if not isinstance(msg, dict):
            continue
        lap_num = msg.get("lap_number")
        if lap_num is not None:
            current_max_lap = max(current_max_lap, int(lap_num))
        else:
            lap_num = current_max_lap

        ln = int(lap_num)
        lap_status.setdefault(ln, {"1"})

        category = str(msg.get("category") or "")
        flag = str(msg.get("flag") or "").upper()
        text = str(msg.get("message") or "").upper()

        if flag == "CHEQUERED" or "CHEQUERED FLAG" in text or "SESSION FINISHED" in text:
            break

        # Check for deleted lap times ("CAR 1 (VER) TIME ... DELETED")
        if "DELETED" in text and "CAR " in text:
            drv_num = msg.get("driver_number")
            if drv_num is not None and msg.get("lap_number") is not None:
                deleted_driver_laps.add((int(drv_num), int(msg["lap_number"])))

        # Track status transitions
        if flag == "RED" or "RED FLAG" in text:
            lap_status[ln].add("5")
        elif category == "SafetyCar" or "SAFETY CAR" in text:
            if "VIRTUAL SAFETY CAR DEPLOYED" in text:
                vsc_active = True
                lap_status[ln].add("6")
            elif "VIRTUAL SAFETY CAR ENDING" in text:
                lap_status[ln].add("7")
                vsc_active = False
            elif "SAFETY CAR DEPLOYED" in text:
                sc_active = True
                lap_status[ln].add("4")
            elif "SAFETY CAR IN THIS LAP" in text:
                lap_status[ln].add("4")
                sc_active = False
        elif flag in ("YELLOW", "DOUBLE YELLOW"):
            lap_status[ln].add("2")
        elif flag in ("GREEN", "CLEAR") and ("TRACK" in text or "GREEN LIGHT" in text):
            sc_active = False
            vsc_active = False

        if sc_active:
            lap_status[ln].add("4")
        if vsc_active:
            lap_status[ln].add("6")

    # Consolidate status string per lap (e.g., '1', '12', '4', '67')
    status_str_map: Dict[int, str] = {}
    for ln, codes in lap_status.items():
        if len(codes) > 1 and "1" in codes:
            codes = codes - {"1"}
        status_str_map[ln] = "".join(sorted(codes))

    return status_str_map, deleted_driver_laps


def _load_via_fastf1_cached_openf1(
    season: int,
    circuit_key: str,
    gp_name: str,
) -> pd.DataFrame:
    """Assemble FastF1-compatible `session.laps` DataFrame cached via `fastf1.Cache`.

    When `livetiming.formula1.com` returns 401 (requiring a paid F1TV subscription in
    FastF1 >= 3.9), this function fetches the identical 2024 FIA LiveTiming streams
    (laps, stints, pit stops, weather, race control, drivers) through
    `fastf1.Cache.requests_get` so all responses are persisted in `data/cache/`.
    """
    sessions = _cached_json_get(f"{OPENF1_BASE_URL}/sessions?year={season}&session_name=Race")
    country_target, loc_target = OPENF1_LOCATION_MAP_2024.get(circuit_key, (circuit_key, circuit_key))

    session_meta = None
    for s in sessions:
        if (
            s.get("country_name") == country_target
            and s.get("location") == loc_target
        ) or s.get("location") == loc_target:
            session_meta = s
            break

    if session_meta is None:
        raise ValueError(f"Could not find {season} Race session for {circuit_key} ({gp_name})")

    sk = int(session_meta["session_key"])
    laps_json = _cached_json_get(f"{OPENF1_BASE_URL}/laps?session_key={sk}")
    stints_json = _cached_json_get(f"{OPENF1_BASE_URL}/stints?session_key={sk}")
    pits_json = _cached_json_get(f"{OPENF1_BASE_URL}/pit?session_key={sk}")
    weather_json = _cached_json_get(f"{OPENF1_BASE_URL}/weather?session_key={sk}")
    rc_json = _cached_json_get(f"{OPENF1_BASE_URL}/race_control?session_key={sk}")
    drivers_json = _cached_json_get(f"{OPENF1_BASE_URL}/drivers?session_key={sk}")

    laps_df = pd.DataFrame(laps_json)
    stints_df = pd.DataFrame(stints_json)
    pits_df = pd.DataFrame(pits_json)
    weather_df = pd.DataFrame(weather_json)
    drivers_df = pd.DataFrame(drivers_json)

    # 1. Driver lookup (Driver acronym & Team name)
    drivers_sub = (
        drivers_df[["driver_number", "name_acronym", "team_name"]]
        .drop_duplicates(subset=["driver_number"])
        .rename(
            columns={
                "driver_number": "DriverNumberInt",
                "name_acronym": "Driver",
                "team_name": "Team",
            }
        )
    )

    laps_df = laps_df.rename(
        columns={
            "driver_number": "DriverNumberInt",
            "lap_number": "LapNumber",
            "lap_duration": "LapTimeSeconds",
            "duration_sector_1": "Sector1Seconds",
            "duration_sector_2": "Sector2Seconds",
            "duration_sector_3": "Sector3Seconds",
            "is_pit_out_lap": "IsOutLapRaw",
        }
    )
    laps_df["DriverNumber"] = laps_df["DriverNumberInt"].astype(str)
    laps_df = laps_df.merge(drivers_sub, on="DriverNumberInt", how="left")

    # Recover LapTimeSeconds if missing but all 3 sectors are present
    missing_lt = laps_df["LapTimeSeconds"].isna()
    has_sectors = (
        laps_df["Sector1Seconds"].notna()
        & laps_df["Sector2Seconds"].notna()
        & laps_df["Sector3Seconds"].notna()
    )
    laps_df.loc[missing_lt & has_sectors, "LapTimeSeconds"] = (
        laps_df.loc[missing_lt & has_sectors, "Sector1Seconds"]
        + laps_df.loc[missing_lt & has_sectors, "Sector2Seconds"]
        + laps_df.loc[missing_lt & has_sectors, "Sector3Seconds"]
    )

    # 2. Expand stints to (DriverNumberInt, LapNumber) -> (Stint, Compound, TyreLife, FreshTyre)
    stint_records: List[Dict[str, object]] = []
    for _, srow in stints_df.sort_values(["driver_number", "stint_number"]).iterrows():
        if pd.isna(srow.get("lap_start")) or pd.isna(srow.get("lap_end")):
            continue
        d_num = int(srow["driver_number"])
        stint_num = int(srow["stint_number"])
        l_start = int(srow["lap_start"])
        l_end = int(srow["lap_end"])
        comp = str(srow.get("compound") or "UNKNOWN").upper()
        age_start = int(srow["tyre_age_at_start"]) if pd.notna(srow.get("tyre_age_at_start")) else 0
        is_fresh = age_start == 0

        for ln in range(l_start, l_end + 1):
            stint_records.append(
                {
                    "DriverNumberInt": d_num,
                    "LapNumber": ln,
                    "Stint": stint_num,
                    "Compound": comp,
                    "TyreLife": age_start + (ln - l_start) + 1,
                    "FreshTyre": is_fresh,
                    "StintStartLapRaw": l_start,
                    "StintEndLapRaw": l_end,
                }
            )

    stint_map_df = (
        pd.DataFrame(stint_records)
        .drop_duplicates(subset=["DriverNumberInt", "LapNumber"], keep="last")
    )
    laps_df = laps_df.merge(stint_map_df, on=["DriverNumberInt", "LapNumber"], how="left")

    # 3. Pit in-laps and out-laps
    in_lap_pairs: Set[Tuple[int, int]] = set()
    pit_duration_map: Dict[Tuple[int, int], float] = {}
    if not pits_df.empty:
        for _, prow in pits_df.iterrows():
            if pd.notna(prow.get("driver_number")) and pd.notna(prow.get("lap_number")):
                pair = (int(prow["driver_number"]), int(prow["lap_number"]))
                in_lap_pairs.add(pair)
                if pd.notna(prow.get("lane_duration")):
                    pit_duration_map[pair] = float(prow["lane_duration"])
                elif pd.notna(prow.get("pit_duration")):
                    pit_duration_map[pair] = float(prow["pit_duration"])

    # Also mark the last lap of any stint (where a subsequent stint exists for that driver) as an in-lap
    max_stint_per_drv = laps_df.groupby("DriverNumberInt")["Stint"].transform("max")
    is_stint_transition_in = (
        (laps_df["LapNumber"] == laps_df["StintEndLapRaw"])
        & (laps_df["Stint"] < max_stint_per_drv)
    )
    is_stint_transition_out = (
        (laps_df["LapNumber"] == laps_df["StintStartLapRaw"])
        & (laps_df["Stint"] > 1)
    )

    laps_df["IsInLap"] = [
        ((int(d), int(l)) in in_lap_pairs) or bool(trans_in)
        for d, l, trans_in in zip(
            laps_df["DriverNumberInt"], laps_df["LapNumber"], is_stint_transition_in
        )
    ]
    laps_df["IsOutLap"] = laps_df["IsOutLapRaw"].fillna(False).astype(bool) | is_stint_transition_out.fillna(False)
    laps_df["PitLaneDurationSeconds"] = [
        pit_duration_map.get((int(d), int(l)), np.nan)
        for d, l in zip(laps_df["DriverNumberInt"], laps_df["LapNumber"])
    ]

    # Provide FastF1-compatible PitInTime / PitOutTime Timedelta columns
    laps_df["LapTime"] = pd.to_timedelta(laps_df["LapTimeSeconds"], unit="s")
    laps_df["Sector1Time"] = pd.to_timedelta(laps_df["Sector1Seconds"], unit="s")
    laps_df["Sector2Time"] = pd.to_timedelta(laps_df["Sector2Seconds"], unit="s")
    laps_df["Sector3Time"] = pd.to_timedelta(laps_df["Sector3Seconds"], unit="s")
    laps_df["PitInTime"] = pd.Series(
        [pd.Timedelta(seconds=1) if flag else pd.NaT for flag in laps_df["IsInLap"]]
    )
    laps_df["PitOutTime"] = pd.Series(
        [pd.Timedelta(seconds=1) if flag else pd.NaT for flag in laps_df["IsOutLap"]]
    )

    # 4. TrackStatus and IsAccurate from FIA Race Control messages
    status_map, deleted_set = _build_neutralized_laps_from_race_control(rc_json)
    laps_df["TrackStatus"] = [
        status_map.get(int(ln), "1") for ln in laps_df["LapNumber"]
    ]
    laps_df["IsNeutralized"] = laps_df["TrackStatus"].apply(_has_interrupted_track_status)
    laps_df["IsLapOne"] = laps_df["LapNumber"] <= 1
    laps_df["Deleted"] = [
        (int(d), int(l)) in deleted_set
        for d, l in zip(laps_df["DriverNumberInt"], laps_df["LapNumber"])
    ]
    laps_df["IsAccurate"] = (
        laps_df["LapTimeSeconds"].notna()
        & (~laps_df["IsLapOne"])
        & (~laps_df["IsInLap"])
        & (~laps_df["IsOutLap"])
        & (~laps_df["IsNeutralized"])
    )

    # 5. Align weather data by timestamp (`date_start` vs weather `date`) using merge_asof
    laps_df["LapStartTimestamp"] = pd.to_datetime(laps_df["date_start"], utc=True, errors="coerce")
    # Fill any missing lap timestamps with session start or interpolation within driver
    laps_df["LapStartTimestamp"] = (
        laps_df.groupby("DriverNumberInt")["LapStartTimestamp"]
        .transform(lambda s: s.ffill().bfill())
    )

    if not weather_df.empty and "date" in weather_df.columns:
        weather_df["WeatherTimestamp"] = pd.to_datetime(weather_df["date"], utc=True, errors="coerce")
        weather_df = (
            weather_df.dropna(subset=["WeatherTimestamp"])
            .sort_values("WeatherTimestamp")
            .rename(
                columns={
                    "air_temperature": "AirTemp",
                    "track_temperature": "TrackTemp",
                    "humidity": "Humidity",
                    "pressure": "Pressure",
                    "rainfall": "Rainfall",
                    "wind_speed": "WindSpeed",
                }
            )
        )
        weather_cols = ["WeatherTimestamp", "AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]
        weather_sub = weather_df[[c for c in weather_cols if c in weather_df.columns]]

        valid_ts_mask = laps_df["LapStartTimestamp"].notna()
        laps_with_ts = laps_df[valid_ts_mask].sort_values("LapStartTimestamp")
        laps_no_ts = laps_df[~valid_ts_mask].copy()

        merged_ts = pd.merge_asof(
            laps_with_ts,
            weather_sub,
            left_on="LapStartTimestamp",
            right_on="WeatherTimestamp",
            direction="nearest",
        )
        if not laps_no_ts.empty:
            for col in ("AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"):
                if col in weather_sub.columns:
                    laps_no_ts[col] = weather_sub[col].median()
            laps_df = pd.concat([merged_ts, laps_no_ts], ignore_index=True)
        else:
            laps_df = merged_ts
    else:
        for col in ("AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"):
            laps_df[col] = np.nan

    # 6. Compute approximate running Position per lap from cumulative elapsed time
    laps_df = laps_df.sort_values(["DriverNumberInt", "LapNumber"]).reset_index(drop=True)
    laps_df["CumTime"] = laps_df.groupby("DriverNumberInt")["LapTimeSeconds"].cumsum()
    laps_df["Position"] = (
        laps_df.groupby("LapNumber")["CumTime"]
        .rank(method="first", na_option="bottom")
        .astype(int)
    )

    # 7. Event metadata
    laps_df["Season"] = season
    laps_df["Circuit"] = circuit_key
    laps_df["EventName"] = gp_name
    laps_df["TotalRaceLaps"] = int(laps_df["LapNumber"].max())

    return laps_df.sort_values(["DriverNumberInt", "LapNumber"]).reset_index(drop=True)


def load_single_race_raw(
    season: int,
    circuit_key: str,
    gp_name: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a single Race ('R') session from FastF1 with weather data attached.

    Uses direct `session.load()` when F1TV credentials (`F1_USERNAME`/`F1_PASSWORD`) or
    local `.ff1pkl` session caches are present; otherwise uses `fastf1.Cache.requests_get`
    against the public FIA LiveTiming mirror (`api.openf1.org/v1/`) so no API key or
    paid subscription is required.
    """
    import os

    target_cache = setup_fastf1_cache(cache_dir)
    fastf1.set_log_level("ERROR")

    has_f1_creds = bool(os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD"))
    has_pkl_cache = any(target_cache.rglob("*.ff1pkl"))

    if has_f1_creds or has_pkl_cache:
        try:
            session = fastf1.get_session(season, gp_name, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            laps = session.laps.copy().reset_index(drop=True)
            if len(laps) > 0:
                weather = session.laps.get_weather_data().copy().reset_index(drop=True)
                for col in ("AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"):
                    if col in weather.columns:
                        laps[col] = weather[col].values

                laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
                laps["Sector1Seconds"] = laps["Sector1Time"].dt.total_seconds()
                laps["Sector2Seconds"] = laps["Sector2Time"].dt.total_seconds()
                laps["Sector3Seconds"] = laps["Sector3Time"].dt.total_seconds()
                laps["IsInLap"] = laps["PitInTime"].notna()
                laps["IsOutLap"] = laps["PitOutTime"].notna()
                laps["IsNeutralized"] = laps["TrackStatus"].apply(_has_interrupted_track_status)
                laps["IsLapOne"] = laps["LapNumber"] <= 1
                laps["Season"] = season
                laps["Circuit"] = circuit_key
                laps["EventName"] = str(session.event["EventName"])
                laps["TotalRaceLaps"] = int(laps["LapNumber"].max())
                return laps
        except Exception as exc:
            logger.warning("Direct session.load fallback triggered for %s: %s", gp_name, exc)

    return _load_via_fastf1_cached_openf1(season, circuit_key, gp_name)


def filter_stint_outliers(df_clean_candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply two-stage outlier filter (107% driver pace + per-stint degradation residual IQR).

    Why this design (Decision A1):
    Standard static cutoffs per stint can mistakenly discard high-degradation laps at the
    end of a long stint. By first removing obvious slow laps (>107% of a driver's fastest
    clean lap in that race) and then detrending each driver-stint linearly (`LapTimeSeconds`
    vs `TyreLife`) before applying a 1.5 * IQR fence on residuals, genuine wear at the end
    of a stint is preserved while traffic spikes and lock-ups are removed.
    """
    df = df_clean_candidates.copy()

    # Stage 1: 107% rule relative to each driver's fastest representative lap in that race
    driver_min_lap = df.groupby(["Circuit", "Driver"])["LapTimeSeconds"].transform("min")
    exceeds_107pct = df["LapTimeSeconds"] > (PACE_THRESHOLD_RATIO * driver_min_lap)

    # Stage 2: Per-stint residual IQR relative to linear tyre degradation trend
    is_residual_outlier = pd.Series(False, index=df.index)

    valid_stage1 = df[~exceeds_107pct]
    for _, group in valid_stage1.groupby(["Circuit", "Driver", "Stint"]):
        idx = group.index
        y = group["LapTimeSeconds"].to_numpy(dtype=float)
        x = group["TyreLife"].to_numpy(dtype=float)

        if len(group) >= MIN_STINT_LAPS_FOR_TREND and np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, deg=1)
            residuals = y - (slope * x + intercept)
        else:
            residuals = y - np.median(y)

        q25, q75 = np.percentile(residuals, [25, 75])
        iqr = max(q75 - q25, 0.15)  # Minimum 0.15s floor so very consistent stints don't over-prune
        lower_fence = q25 - STINT_RESIDUAL_IQR_MULTIPLIER * iqr
        upper_fence = q75 + STINT_RESIDUAL_IQR_MULTIPLIER * iqr

        outlier_mask = (residuals < lower_fence) | (residuals > upper_fence)
        is_residual_outlier.loc[idx] = outlier_mask

    is_outlier = exceeds_107pct | is_residual_outlier
    return df[~is_outlier].copy(), is_outlier


def clean_race_laps(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter non-representative laps and produce both cleaned dataset and audit summary."""
    df = raw_df.copy()

    has_basic_fields = (
        df["LapTimeSeconds"].notna()
        & df["TyreLife"].notna()
        & df["Compound"].isin(DRY_COMPOUNDS)
    )
    not_lap1 = ~df["IsLapOne"]
    not_pit = (~df["IsInLap"]) & (~df["IsOutLap"])
    not_neutralized = ~df["IsNeutralized"]
    is_accurate = df["IsAccurate"].fillna(False).astype(bool)

    candidate_mask = (
        has_basic_fields & not_lap1 & not_pit & not_neutralized & is_accurate
    )
    candidates = df[candidate_mask].copy()

    clean_df, outlier_series = filter_stint_outliers(candidates)
    df["IsCandidate"] = candidate_mask
    df["IsStintOutlier"] = False
    df.loc[outlier_series.index, "IsStintOutlier"] = outlier_series

    # Also compute clean laps and stints per compound for each circuit
    audit_rows: List[Dict[str, object]] = []
    for circuit, grp in df.groupby("Circuit", sort=False):
        n_total = len(grp)
        n_lap1 = int(grp["IsLapOne"].sum())
        n_pit = int((grp["IsInLap"] | grp["IsOutLap"]).sum())
        n_sc_vsc = int(grp["IsNeutralized"].sum())
        n_rain = int((grp["Rainfall"].fillna(0).astype(float) > 0).sum()) if "Rainfall" in grp else 0
        n_candidates = int(grp["IsCandidate"].sum())
        n_outliers = int(grp["IsStintOutlier"].sum())
        n_clean = n_candidates - n_outliers

        c_clean = clean_df[clean_df["Circuit"] == circuit]
        stint_counts = (
            c_clean.groupby("Compound")[["Driver", "Stint"]]
            .apply(lambda g: len(g.drop_duplicates()))
            .to_dict()
        )
        lap_counts = c_clean["Compound"].value_counts().to_dict()

        audit_rows.append(
            {
                "Circuit": circuit,
                "EventName": grp["EventName"].iloc[0],
                "TotalRaceLaps": int(grp["TotalRaceLaps"].iloc[0]),
                "RawLaps": n_total,
                "Lap1Laps": n_lap1,
                "PitInOutLaps": n_pit,
                "SC_VSC_RedLaps": n_sc_vsc,
                "SC_VSC_Pct": round(100.0 * n_sc_vsc / max(n_total, 1), 2),
                "RainLaps": n_rain,
                "CandidateCleanLaps": n_candidates,
                "StintOutliersRemoved": n_outliers,
                "FinalCleanLaps": n_clean,
                "CleanRetentionPct": round(100.0 * n_clean / max(n_total, 1), 2),
                "SoftStints": int(stint_counts.get("SOFT", 0)),
                "MediumStints": int(stint_counts.get("MEDIUM", 0)),
                "HardStints": int(stint_counts.get("HARD", 0)),
                "SoftCleanLaps": int(lap_counts.get("SOFT", 0)),
                "MediumCleanLaps": int(lap_counts.get("MEDIUM", 0)),
                "HardCleanLaps": int(lap_counts.get("HARD", 0)),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    return clean_df.reset_index(drop=True), audit_df


def estimate_circuit_pit_loss(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate empirical pit stop time loss per circuit from actual race pit stops.

    Formula required by project specification:
        PitLoss = median( [in_lap + out_lap] - 2 * driver_normal_lap_time )
    """
    driver_normal = (
        clean_df.groupby(["Circuit", "Driver"])["LapTimeSeconds"]
        .median()
        .rename("DriverNormalLapTime")
        .reset_index()
    )

    raw_sorted = raw_df.sort_values(["Circuit", "Driver", "LapNumber"]).reset_index(drop=True)
    raw_sorted = raw_sorted.merge(driver_normal, on=["Circuit", "Driver"], how="left")

    pit_events: List[Dict[str, object]] = []

    for (circuit, driver), grp in raw_sorted.groupby(["Circuit", "Driver"], sort=False):
        grp = grp.sort_values("LapNumber").reset_index(drop=True)
        driver_clean = clean_df[
            (clean_df["Circuit"] == circuit) & (clean_df["Driver"] == driver)
        ]

        for i in range(len(grp) - 1):
            in_row = grp.iloc[i]
            out_row = grp.iloc[i + 1]

            if not (bool(in_row["IsInLap"]) and bool(out_row["IsOutLap"])):
                continue
            if int(out_row["LapNumber"]) != int(in_row["LapNumber"]) + 1:
                continue
            if bool(in_row["IsNeutralized"]) or bool(out_row["IsNeutralized"]):
                continue
            if int(in_row["LapNumber"]) <= 1:
                continue

            in_time = float(in_row["LapTimeSeconds"]) if pd.notna(in_row["LapTimeSeconds"]) else np.nan
            out_time = float(out_row["LapTimeSeconds"]) if pd.notna(out_row["LapTimeSeconds"]) else np.nan
            normal_time = (
                float(in_row["DriverNormalLapTime"])
                if pd.notna(in_row["DriverNormalLapTime"])
                else np.nan
            )

            if np.isnan(in_time) or np.isnan(out_time) or np.isnan(normal_time):
                continue

            pit_lap = int(in_row["LapNumber"])
            nearby_clean = driver_clean[
                driver_clean["LapNumber"].between(pit_lap - 4, pit_lap + 5)
            ]["LapTimeSeconds"]
            local_normal = float(nearby_clean.median()) if len(nearby_clean) >= 2 else normal_time

            pit_loss_race_norm = (in_time + out_time) - 2.0 * normal_time
            pit_loss_local_norm = (in_time + out_time) - 2.0 * local_normal

            is_valid_stop = PIT_LOSS_MIN_SECONDS <= pit_loss_race_norm <= PIT_LOSS_MAX_SECONDS

            pit_events.append(
                {
                    "Circuit": circuit,
                    "Driver": driver,
                    "Team": in_row["Team"],
                    "InLapNumber": pit_lap,
                    "OutLapNumber": int(out_row["LapNumber"]),
                    "OldCompound": in_row["Compound"],
                    "NewCompound": out_row["Compound"],
                    "InLapSeconds": round(in_time, 3),
                    "OutLapSeconds": round(out_time, 3),
                    "DriverNormalLapSeconds": round(normal_time, 3),
                    "LocalNormalLapSeconds": round(local_normal, 3),
                    "PitLossSeconds": round(pit_loss_race_norm, 3),
                    "PitLossLocalSeconds": round(pit_loss_local_norm, 3),
                    "IsValidGreenStop": is_valid_stop,
                }
            )

    pit_events_df = pd.DataFrame(pit_events)
    valid_events = pit_events_df[pit_events_df["IsValidGreenStop"]].copy()

    summary_rows: List[Dict[str, object]] = []
    for circuit in raw_df["Circuit"].unique():
        c_events = valid_events[valid_events["Circuit"] == circuit]
        losses = c_events["PitLossSeconds"].to_numpy(dtype=float)
        local_losses = c_events["PitLossLocalSeconds"].to_numpy(dtype=float)

        if len(losses) > 0:
            q25, q75 = np.percentile(losses, [25, 75])
            summary_rows.append(
                {
                    "Circuit": circuit,
                    "ValidPitStops": int(len(losses)),
                    "TotalGreenPitStops": int(
                        len(pit_events_df[pit_events_df["Circuit"] == circuit])
                    ),
                    "MedianPitLossSeconds": round(float(np.median(losses)), 2),
                    "MeanPitLossSeconds": round(float(np.mean(losses)), 2),
                    "StdPitLossSeconds": round(float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0, 2),
                    "IQRPitLossSeconds": round(float(q75 - q25), 2),
                    "MinPitLossSeconds": round(float(np.min(losses)), 2),
                    "MaxPitLossSeconds": round(float(np.max(losses)), 2),
                    "MedianLocalPitLossSeconds": round(float(np.median(local_losses)), 2),
                }
            )

    pit_summary_df = pd.DataFrame(summary_rows)
    return pit_summary_df, pit_events_df


def extract_driver_stint_timeline(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize every driver stint per race for EDA stint-length analysis & strategy timelines."""
    valid_raw = raw_df[raw_df["Compound"].isin(DRY_COMPOUNDS)].copy()

    driver_final = (
        valid_raw.sort_values(["Circuit", "LapNumber"])
        .groupby(["Circuit", "Driver"], as_index=False)
        .last()[["Circuit", "Driver", "Position", "LapNumber"]]
        .rename(columns={"Position": "FinalPosition", "LapNumber": "DriverTotalLaps"})
    )

    stints = (
        valid_raw.groupby(["Circuit", "EventName", "Driver", "Team", "Stint", "Compound"], as_index=False)
        .agg(
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            TotalStintLaps=("LapNumber", "count"),
            InitialTyreLife=("TyreLife", "min"),
            MaxTyreLife=("TyreLife", "max"),
            FreshTyre=("FreshTyre", "first"),
        )
    )

    clean_counts = (
        clean_df.groupby(["Circuit", "Driver", "Stint"], as_index=False)
        .agg(
            CleanStintLaps=("LapNumber", "count"),
            MedianLapTimeSeconds=("LapTimeSeconds", "median"),
        )
    )

    stints = stints.merge(clean_counts, on=["Circuit", "Driver", "Stint"], how="left")
    stints["CleanStintLaps"] = stints["CleanStintLaps"].fillna(0).astype(int)
    stints = stints.merge(driver_final, on=["Circuit", "Driver"], how="left")
    stints = stints.sort_values(["Circuit", "FinalPosition", "Driver", "Stint"]).reset_index(drop=True)
    return stints


def load_all_races(
    races: Optional[Dict[str, str]] = None,
    season: int = 2024,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and concatenate raw lap data for all configured races."""
    target_races = races or DEFAULT_RACES_2024
    frames: List[pd.DataFrame] = []
    for circuit_key, gp_name in target_races.items():
        logger.info("Loading %d %s (%s)...", season, gp_name, circuit_key)
        df_race = load_single_race_raw(season, circuit_key, gp_name, cache_dir=cache_dir)
        frames.append(df_race)
    return pd.concat(frames, ignore_index=True)


def run_stage2_pipeline(
    races: Optional[Dict[str, str]] = None,
    season: int = 2024,
    save_processed: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Execute complete Stage 2 loading, cleaning, pit-loss estimation, and feature engineering."""
    from src.features import build_features

    raw_df = load_all_races(races=races, season=season)
    clean_df, audit_df = clean_race_laps(raw_df)
    pit_summary_df, pit_events_df = estimate_circuit_pit_loss(raw_df, clean_df)
    stints_df = extract_driver_stint_timeline(raw_df, clean_df)
    feat_df, fuel_calib_df = build_features(clean_df)

    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(PROCESSED_DIR / "clean_laps_2024.csv", index=False)
        audit_df.to_csv(PROCESSED_DIR / "race_audit_2024.csv", index=False)
        pit_summary_df.to_csv(PROCESSED_DIR / "pit_loss_summary_2024.csv", index=False)
        pit_events_df.to_csv(PROCESSED_DIR / "pit_events_2024.csv", index=False)
        stints_df.to_csv(PROCESSED_DIR / "driver_stints_2024.csv", index=False)
        fuel_calib_df.to_csv(PROCESSED_DIR / "fuel_calibration_2024.csv", index=False)

    return {
        "raw_laps": raw_df,
        "clean_laps": feat_df,
        "audit": audit_df,
        "pit_summary": pit_summary_df,
        "pit_events": pit_events_df,
        "stints": stints_df,
        "fuel_calibration": fuel_calib_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = run_stage2_pipeline()
    print("\n=== RACE DATA QUALITY AUDIT ===")
    print(results["audit"].to_string(index=False))
    print("\n=== EMPIRICAL PIT STOP LOSS PER CIRCUIT ===")
    print(results["pit_summary"].to_string(index=False))
    print("\n=== FUEL CORRECTION CALIBRATION ===")
    print(results["fuel_calibration"].to_string(index=False))
