#!/usr/bin/env python3
"""
Masters performance regression using Data Golf API.

Training target (Y): Masters finish from pre-tournament-archive (fin_text).

Features (X) — tournament-based (always in the model):
  - Prior ~6 months of PGA Tour events before each Masters (pre-tournament-archive).
  - Prior Masters history: decay-weighted mean finish, best in a recent window, starts.
  - Skill-table age at Masters (back-cast on training rows; direct age for current projection).
  - Low 6m PGA sample: pessimistic form constants + pga6m_insufficient flag (see flags).

Optional skill metrics (current API snapshots): merged when using
  --include-skill-features (fixed bundle) or --skill-forward-select (iterative).

--skill-forward-select runs forward selection: each candidate must pass
  (1) full-sample partial F with p < --entry-alpha, and
  (2) leave-one-Masters-year-out (LOYO) mean RMSE improvement vs the reduced model.
Forward selection uses classical OLS for gates; the final reported fit uses
cluster-robust SE by dg_id when possible. Stepwise search is exploratory.

API key: DATAGOLF_KEY in the environment.

Usage:
  export DATAGOLF_KEY=your_key
  pip install -r requirements.txt
  python masters_regression.py
  python masters_regression.py --skill-forward-select --entry-alpha 0.05
  python masters_regression.py --include-skill-features
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
import scipy.stats as scipy_stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


BASE = "https://feeds.datagolf.com"

MASTERS_ARCHIVE = [
    (2020, 14),
    (2021, 536),
    (2022, 14),
    (2023, 14),
    (2024, 14),
    (2025, 14),
]

# Ordered skill candidates for --skill-forward-select (theory-ish: approach/putting first).
# Only columns present after merge_skill_features are used.
SKILL_CANDIDATE_ORDER: list[str] = [
    "sg_app",
    "sg_putt",
    "sg_arg",
    "sg_ott",
    "driving_dist",
    "driving_acc",
    "gir_pct",
    "par3_approach_proxy",
    "par5_long_approach_proxy",
    "course_history",
    "timing_adjustment",
    "short_game_tight_lie_proxy",
]


@dataclass
class FetchConfig:
    key: str
    min_interval_s: float = 1.35

    def __post_init__(self) -> None:
        self._last_req: float = 0.0

    def get(self, path: str, params: dict[str, Any]) -> Any:
        now = time.monotonic()
        wait = self.min_interval_s - (now - self._last_req)
        if wait > 0:
            time.sleep(wait)
        p = {**params, "key": self.key}
        url = f"{BASE}{path}"
        r = requests.get(url, params=p, timeout=120)
        self._last_req = time.monotonic()
        if r.status_code == 403:
            print(f"403 from {path}: {r.text[:200]}", file=sys.stderr)
        r.raise_for_status()
        return r.json()


def parse_finish(fin: str | None) -> float | None:
    if fin is None or (isinstance(fin, float) and np.isnan(fin)):
        return None
    s = str(fin).strip().upper()
    if s in {"CUT", "WD", "DQ", "MD", "DNS"}:
        return None
    if s.startswith("T"):
        s = s[1:]
    try:
        return float(int(s))
    except ValueError:
        return None


def fetch_archive_year(fc: FetchConfig, year: int, event_id: int) -> pd.DataFrame:
    js = fc.get(
        "/preds/pre-tournament-archive",
        {"event_id": event_id, "year": year, "file_format": "json"},
    )
    rows = js.get("baseline") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["season_year"] = year
    df["finish_rank"] = df["fin_text"].map(parse_finish)
    return df


def fetch_archive_safe(
    fc: FetchConfig, event_id: int, year: int
) -> pd.DataFrame | None:
    try:
        js = fc.get(
            "/preds/pre-tournament-archive",
            {"event_id": event_id, "year": year, "file_format": "json"},
        )
    except (requests.HTTPError, requests.JSONDecodeError, ValueError) as e:
        print(f"  skip archive event_id={event_id} year={year}: {e}", file=sys.stderr)
        return None
    rows = js.get("baseline") or []
    if not rows:
        return None
    return pd.DataFrame(rows)


def fetch_pga_event_list(fc: FetchConfig) -> list[dict[str, Any]]:
    js = fc.get("/historical-raw-data/event-list", {"tour": "pga", "file_format": "json"})
    if not isinstance(js, list):
        raise ValueError("Unexpected event-list shape")
    return js


def fetch_schedule_masters_2026(fc: FetchConfig) -> tuple[int, str] | None:
    try:
        js = fc.get(
            "/get-schedule",
            {"tour": "pga", "season": "2026", "upcoming_only": "no", "file_format": "json"},
        )
    except requests.HTTPError:
        return None
    for e in js.get("schedule", []):
        if "Masters Tournament" in e.get("event_name", ""):
            return int(e["event_id"]), e["start_date"]
    return None


def masters_rows_from_event_list(event_list: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in event_list:
        name = e.get("event_name", "")
        if "Masters" not in name:
            continue
        if "Women" in name or "women" in name:
            continue
        rows.append(
            {
                "calendar_year": int(e["calendar_year"]),
                "event_id": int(e["event_id"]),
                "date": e["date"],
            }
        )
    return pd.DataFrame(rows).sort_values("date")


def six_month_window_before(masters_date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(masters_date_str)
    start = end - pd.DateOffset(months=6)
    return start, end


def pga_events_in_window(
    event_list: list[dict[str, Any]],
    window_start: pd.Timestamp,
    window_end_excl: pd.Timestamp,
) -> list[dict[str, Any]]:
    out = []
    for e in event_list:
        if e.get("tour") != "pga":
            continue
        name = e.get("event_name", "")
        if "Masters" in name:
            continue
        dt = pd.Timestamp(e["date"])
        if window_start <= dt < window_end_excl:
            out.append(e)
    return out


def aggregate_event_finishes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for df in dfs:
        if df is None or df.empty:
            continue
        if "dg_id" not in df.columns:
            continue
        t = df[["dg_id", "fin_text"]].copy()
        t["finish_rank"] = t["fin_text"].map(parse_finish)
        t["made_cut_numeric"] = t["finish_rank"].notna().astype(float)
        t["top25"] = (t["finish_rank"].notna() & (t["finish_rank"] <= 25)).astype(float)
        parts.append(t[["dg_id", "finish_rank", "made_cut_numeric", "top25"]])
    if not parts:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "pga6m_starts",
                "pga6m_mean_finish",
                "pga6m_median_finish",
                "pga6m_top25_rate",
                "pga6m_top25_count",
                "pga6m_made_cut_rate",
            ]
        )
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "pga6m_starts",
                "pga6m_mean_finish",
                "pga6m_median_finish",
                "pga6m_top25_rate",
                "pga6m_top25_count",
                "pga6m_made_cut_rate",
            ]
        )
    all_rows = pd.concat(parts, ignore_index=True)
    g = all_rows.groupby("dg_id", as_index=False)
    agg = g.agg(
        pga6m_starts=("finish_rank", "count"),
        pga6m_mean_finish=("finish_rank", "mean"),
        pga6m_median_finish=("finish_rank", "median"),
        pga6m_top25_count=("top25", "sum"),
        pga6m_made_cut_rate=("made_cut_numeric", "mean"),
    )
    agg["pga6m_top25_rate"] = agg["pga6m_top25_count"] / agg["pga6m_starts"].clip(lower=1)
    return agg


def prior_masters_features_decay(
    mhist: pd.DataFrame,
    target_year: int,
    decay_lambda: float,
    recent_best_years: int,
) -> pd.DataFrame:
    """
    Prior Masters summaries with exponential decay (recent years weigh more) and
    best finish restricted to the last `recent_best_years` calendar years before target_year.
    """
    cols = [
        "dg_id",
        "masters_prior_starts",
        "masters_prior_decay_mean_finish",
        "masters_prior_best_recent",
    ]
    prev = mhist[
        (mhist["masters_year"] < target_year) & mhist["masters_finish"].notna()
    ].copy()
    if prev.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for dg_id, g in prev.groupby("dg_id", sort=False):
        years = g["masters_year"].to_numpy(dtype=float)
        f = g["masters_finish"].to_numpy(dtype=float)
        lag = target_year - years
        w = np.exp(-decay_lambda * np.maximum(lag, 0.0))
        decay_mean = float(np.dot(w, f) / np.maximum(w.sum(), 1e-12))
        recent_mask = years >= (target_year - recent_best_years)
        best_recent = float(f[recent_mask].min()) if recent_mask.any() else float("nan")
        rows.append(
            {
                "dg_id": dg_id,
                "masters_prior_starts": float(len(f)),
                "masters_prior_decay_mean_finish": decay_mean,
                "masters_prior_best_recent": best_recent,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def build_masters_history_long(
    fc: FetchConfig, cache: dict[tuple[int, int], pd.DataFrame | None]
) -> pd.DataFrame:
    frames = []
    for year, eid in MASTERS_ARCHIVE:
        key = (eid, year)
        if key not in cache:
            cache[key] = fetch_archive_safe(fc, eid, year)
        df = cache[key]
        if df is None or df.empty:
            continue
        t = df[["dg_id", "fin_text"]].copy()
        t["masters_year"] = year
        t["masters_finish"] = t["fin_text"].map(parse_finish)
        frames.append(t[["dg_id", "masters_year", "masters_finish"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_age_at_masters(
    df: pd.DataFrame,
    skill: pd.DataFrame,
    ref_year: int,
    season_col: str = "season_year",
    use_direct_age: bool = False,
) -> pd.DataFrame:
    """
    Approximate age at that Masters using skill table `age`.
    Training rows: age_at_masters ≈ age - (ref_year - season_year).
    Current-year projection: set use_direct_age=True so age_at_masters = age.
    """
    out = df.copy()
    if "age" not in skill.columns or "dg_id" not in skill.columns:
        out["age_at_masters"] = np.nan
        return out
    age_sub = skill[["dg_id", "age"]].drop_duplicates(subset=["dg_id"])
    out = out.merge(age_sub, on="dg_id", how="left")
    age_num = pd.to_numeric(out["age"], errors="coerce")
    if use_direct_age:
        out["age_at_masters"] = age_num
    else:
        sy = pd.to_numeric(out[season_col], errors="coerce")
        out["age_at_masters"] = age_num - (float(ref_year) - sy)
    out.drop(columns=["age"], errors="ignore", inplace=True)
    return out


def apply_pga6m_insufficient_form(
    df: pd.DataFrame,
    min_starts: int,
    pessimistic_mean_finish: float = 65.0,
    pessimistic_top25_rate: float = 0.0,
    pessimistic_made_cut_rate: float = 0.35,
) -> None:
    """
    In-place: set pga6m_insufficient flag; overwrite form metrics for low-sample rows
    so we do not median-impute 'average tour form' for part-time players.
    """
    starts = pd.to_numeric(df["pga6m_starts"], errors="coerce").fillna(0.0)
    df["pga6m_insufficient"] = (starts < float(min_starts)).astype(float)
    ins = df["pga6m_insufficient"] > 0
    if ins.any():
        df.loc[ins, "pga6m_mean_finish"] = pessimistic_mean_finish
        df.loc[ins, "pga6m_top25_rate"] = pessimistic_top25_rate
        df.loc[ins, "pga6m_made_cut_rate"] = pessimistic_made_cut_rate


def collect_needed_archives(
    fc: FetchConfig,
    event_list: list[dict[str, Any]],
    masters_meta: pd.DataFrame,
    extra_masters: tuple[int, str] | None,
) -> dict[tuple[int, int], pd.DataFrame | None]:
    cache: dict[tuple[int, int], pd.DataFrame | None] = {}

    for year, eid in MASTERS_ARCHIVE:
        cache[(eid, year)] = fetch_archive_safe(fc, eid, year)

    for _, mr in masters_meta.iterrows():
        cy = int(mr["calendar_year"])
        mdate = str(mr["date"])
        eid_m = int(mr["event_id"])
        w0, w1 = six_month_window_before(mdate)
        evs = pga_events_in_window(event_list, w0, w1)
        print(
            f"  Masters {cy} (id {eid_m} @ {mdate}): "
            f"{len(evs)} PGA events in prior 6 months"
        )
        for e in evs:
            key = (int(e["event_id"]), int(e["calendar_year"]))
            if key in cache:
                continue
            cache[key] = fetch_archive_safe(fc, key[0], key[1])

    if extra_masters:
        eid, dstr = extra_masters
        w0, w1 = six_month_window_before(dstr)
        evs = pga_events_in_window(event_list, w0, w1)
        print(
            f"  2026 Masters projection window @ {dstr}: {len(evs)} PGA events in prior 6 months"
        )
        for e in evs:
            key = (int(e["event_id"]), int(e["calendar_year"]))
            if key in cache:
                continue
            cache[key] = fetch_archive_safe(fc, key[0], key[1])

    return cache


def form_features_for_year(
    event_list: list[dict[str, Any]],
    masters_meta: pd.DataFrame,
    target_calendar_year: int,
    archive_cache: dict[tuple[int, int], pd.DataFrame | None],
) -> pd.DataFrame:
    row = masters_meta[masters_meta["calendar_year"] == target_calendar_year]
    if row.empty:
        return pd.DataFrame()
    mdate = str(row.iloc[0]["date"])
    w0, w1 = six_month_window_before(mdate)
    evs = pga_events_in_window(event_list, w0, w1)
    dfs = []
    for e in evs:
        key = (int(e["event_id"]), int(e["calendar_year"]))
        dfs.append(archive_cache.get(key))
    return aggregate_event_finishes(dfs)


def fetch_skill_tables(fc: FetchConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sk = fc.get("/preds/skill-ratings", {"display": "value", "file_format": "json"})
    skill = pd.DataFrame(sk["players"])
    skill["sg_t2g"] = skill["sg_ott"] + skill["sg_app"] + skill["sg_arg"]

    ap = fc.get("/preds/approach-skill", {"period": "l12", "file_format": "json"})
    approach = pd.DataFrame(ap["data"])

    dec = fc.get("/preds/player-decompositions", {"tour": "pga", "file_format": "json"})
    decomp = pd.DataFrame(dec["players"])
    return skill, approach, decomp


def fetch_field(fc: FetchConfig) -> pd.DataFrame:
    fu = fc.get("/field-updates", {"tour": "pga", "file_format": "json"})
    rows = fu.get("field") or fu.get("players") or []
    return pd.DataFrame(rows)


def fetch_live_pre_tournament(fc: FetchConfig) -> pd.DataFrame:
    js = fc.get("/preds/pre-tournament", {"tour": "pga", "file_format": "json"})
    rows = js.get("baseline") or []
    return pd.DataFrame(rows)


def merge_skill_features(
    base: pd.DataFrame,
    skill: pd.DataFrame,
    approach: pd.DataFrame,
    decomp: pd.DataFrame,
) -> pd.DataFrame:
    s_cols = [
        "dg_id",
        "sg_ott",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "sg_t2g",
        "driving_dist",
        "driving_acc",
    ]
    skill_sub = skill[[c for c in s_cols if c in skill.columns]].copy()
    # GIR: common names from Data Golf / traditional stats feeds
    for gir_alias in ("gir", "gir_pct", "gir_rate"):
        if gir_alias in skill.columns:
            skill_sub["gir_pct"] = skill[gir_alias]
            break

    ap_cols = ["dg_id", "over_200_fw_sg_per_shot", "under_150_rgh_sg_per_shot"]
    for short in ("50_100_fw_sg_per_shot", "100_150_fw_sg_per_shot"):
        if short in approach.columns:
            ap_cols.append(short)
    ap_sub = approach[[c for c in ap_cols if c in approach.columns]].copy()

    d_cols = ["dg_id", "total_course_history_adjustment", "timing_adjustment"]
    d_sub = decomp[[c for c in d_cols if c in decomp.columns]].copy()

    out = base.merge(skill_sub, on="dg_id", how="left")
    out = out.merge(ap_sub, on="dg_id", how="left")
    out = out.merge(d_sub, on="dg_id", how="left")

    p3_cols = [c for c in ("50_100_fw_sg_per_shot", "100_150_fw_sg_per_shot") if c in out.columns]
    if p3_cols:
        out["par3_approach_proxy"] = out[p3_cols].mean(axis=1, skipna=True)
    else:
        out["par3_approach_proxy"] = np.nan

    out["short_game_tight_lie_proxy"] = (
        out["under_150_rgh_sg_per_shot"].fillna(0) + out["sg_arg"].fillna(0)
    )
    if "total_course_history_adjustment" in out.columns:
        out["course_history"] = out["total_course_history_adjustment"].fillna(0.0)
    else:
        out["course_history"] = 0.0
    if "driving_dist" in out.columns:
        out["length_proxy"] = out["driving_dist"]
    else:
        out["length_proxy"] = np.nan
    if "over_200_fw_sg_per_shot" in out.columns:
        out["par5_long_approach_proxy"] = out["over_200_fw_sg_per_shot"]
    else:
        out["par5_long_approach_proxy"] = np.nan
    if "driving_acc" in out.columns:
        out["driving_acc"] = out["driving_acc"].astype(float)
    return out


def partial_f_test(
    y: np.ndarray,
    X_full: pd.DataFrame | np.ndarray,
    X_reduced: pd.DataFrame | np.ndarray,
    add_const_full: bool = True,
    add_const_reduced: bool = True,
) -> tuple[float, float, int]:
    if add_const_full:
        Xf = sm.add_constant(X_full, has_constant="add")
    else:
        Xf = X_full
    if add_const_reduced:
        Xr = sm.add_constant(X_reduced, has_constant="add")
    else:
        Xr = X_reduced

    full: RegressionResultsWrapper = sm.OLS(y, Xf).fit()
    red: RegressionResultsWrapper = sm.OLS(y, Xr).fit()

    rss_f = full.ssr
    rss_r = red.ssr
    df_f = int(full.df_resid)
    q = Xf.shape[1] - Xr.shape[1]
    if q <= 0:
        raise ValueError("Reduced model must drop at least one regressor.")
    f_stat = ((rss_r - rss_f) / q) / (rss_f / df_f) if rss_f > 0 else np.nan
    p_val = float(scipy_stats.f.sf(f_stat, q, df_f))
    return float(f_stat), p_val, q


def nested_ols_compare(
    work: pd.DataFrame,
    y_col: str,
    cols_reduced: list[str],
    cols_full: list[str],
) -> dict[str, Any] | None:
    """Full-sample nested OLS: reduced vs full (one extra block in full)."""
    cols_needed = list(dict.fromkeys(cols_reduced + cols_full))
    d = work.dropna(subset=[y_col] + cols_needed).copy()
    if len(d) < len(cols_full) + 3:
        return None
    y = d[y_col].to_numpy(dtype=float)
    Xr = d[cols_reduced]
    Xf = d[cols_full]
    f_stat, p_val, q = partial_f_test(y, Xf, Xr, add_const_full=True, add_const_reduced=True)
    red = sm.OLS(y, sm.add_constant(Xr, has_constant="add")).fit()
    full = sm.OLS(y, sm.add_constant(Xf, has_constant="add")).fit()
    df2 = int(full.df_resid)
    return {
        "F": f_stat,
        "p_value": p_val,
        "df_num": q,
        "df_den": df2,
        "delta_r2": float(full.rsquared - red.rsquared),
        "aic_reduced": float(red.aic),
        "aic_full": float(full.aic),
        "bic_reduced": float(red.bic),
        "bic_full": float(full.bic),
        "n": len(d),
    }


def loyo_mean_rmse(
    df: pd.DataFrame,
    y_col: str,
    cols_reduced: list[str],
    cols_full: list[str],
    group_col: str = "season_year",
    min_train_rows: int | None = None,
) -> tuple[float, float, int, list[tuple[Any, float, float]]]:
    """
    Leave-one-group-out: for each fold, fit OLS on other rows, RMSE on held-out.
    Returns (mean_rmse_reduced, mean_rmse_full, n_folds_used, per_fold_details).
    """
    cols_needed = list(dict.fromkeys(cols_reduced + cols_full))
    d = df.dropna(subset=[y_col] + cols_needed).copy()
    years = sorted(d[group_col].unique())
    if min_train_rows is None:
        min_train_rows = len(cols_full) + 5

    fold_rmses_red: list[float] = []
    fold_rmses_full: list[float] = []
    details: list[tuple[Any, float, float]] = []

    for yv in years:
        tr = d[d[group_col] != yv]
        te = d[d[group_col] == yv]
        if len(tr) < min_train_rows or len(te) == 0:
            continue
        y_tr = tr[y_col].to_numpy(dtype=float)
        y_te = te[y_col].to_numpy(dtype=float)
        Xr_tr = sm.add_constant(tr[cols_reduced], has_constant="add")
        Xf_tr = sm.add_constant(tr[cols_full], has_constant="add")
        try:
            fit_r = sm.OLS(y_tr, Xr_tr).fit()
            fit_f = sm.OLS(y_tr, Xf_tr).fit()
        except Exception:
            continue
        Xr_te = sm.add_constant(te[cols_reduced], has_constant="add")
        Xf_te = sm.add_constant(te[cols_full], has_constant="add")
        pr = fit_r.predict(Xr_te)
        pf = fit_f.predict(Xf_te)
        rmse_r = float(np.sqrt(np.mean((y_te - pr) ** 2)))
        rmse_f = float(np.sqrt(np.mean((y_te - pf) ** 2)))
        fold_rmses_red.append(rmse_r)
        fold_rmses_full.append(rmse_f)
        details.append((yv, rmse_r, rmse_f))

    if not fold_rmses_red:
        return float("nan"), float("nan"), 0, details

    return (
        float(np.mean(fold_rmses_red)),
        float(np.mean(fold_rmses_full)),
        len(fold_rmses_red),
        details,
    )


def forward_select_skills(
    df: pd.DataFrame,
    y_col: str,
    tournament_cols: list[str],
    candidate_order: list[str],
    alpha: float,
    loyo_epsilon: float,
    group_col: str = "season_year",
) -> tuple[list[str], list[dict[str, Any]], set[str], list[str]]:
    """
    Add at most one skill per round. Among candidates with p < alpha (sorted by p),
    first that passes LOYO is admitted. Failed LOYO candidates are removed for the run.
    LOYO-failed candidates are blacklisted. Candidates with p >= alpha are retried
    after another variable is admitted (p can change). Stops when no p < alpha
    candidates or no LOYO pass after trying all significant candidates in a round.

    Returns: (selected, steps, loyo_failed, present_candidates).
    """
    present = [
        c for c in candidate_order if c in df.columns and df[c].notna().any()
    ]
    selected: list[str] = []
    loyo_failed: set[str] = set()
    steps: list[dict[str, Any]] = []
    step_idx = 0

    print(
        "\n=== Forward skill selection (Gate A: p < alpha; Gate B: LOYO mean RMSE) ===\n"
        "Classical OLS for gates; exploratory stepwise. See MODELING.md.\n",
        file=sys.stderr,
    )

    while True:
        pool = [c for c in present if c not in loyo_failed and c not in selected]
        if not pool:
            break

        stats_list: list[tuple[float, str, dict[str, Any]]] = []

        for c in pool:
            cols_red = tournament_cols + selected
            cols_full = cols_red + [c]
            comp = nested_ols_compare(df, y_col, cols_red, cols_full)
            if comp is None:
                continue
            if comp["p_value"] >= alpha:
                continue
            stats_list.append((comp["p_value"], c, comp))

        stats_list.sort(key=lambda x: x[0])

        if not stats_list:
            print("  No candidate with p < alpha; stopping.", file=sys.stderr)
            break

        round_admitted = False
        for _p_val, c, comp in stats_list:
            cols_red = tournament_cols + selected
            cols_full = cols_red + [c]
            rmse_r, rmse_f, n_folds, _fold_detail = loyo_mean_rmse(
                df, y_col, cols_red, cols_full, group_col=group_col
            )
            loyo_ok = (
                not np.isnan(rmse_r)
                and not np.isnan(rmse_f)
                and rmse_f <= rmse_r - loyo_epsilon
            )
            step_idx += 1
            row = {
                "step": step_idx,
                "candidate": c,
                "admitted": loyo_ok,
                "F": comp["F"],
                "df_num": comp["df_num"],
                "df_den": comp["df_den"],
                "p_value": comp["p_value"],
                "delta_r2": comp["delta_r2"],
                "aic_reduced": comp["aic_reduced"],
                "aic_full": comp["aic_full"],
                "bic_reduced": comp["bic_reduced"],
                "bic_full": comp["bic_full"],
                "loyo_rmse_reduced": rmse_r,
                "loyo_rmse_full": rmse_f,
                "loyo_delta": rmse_f - rmse_r if not np.isnan(rmse_f) else float("nan"),
                "loyo_n_folds": n_folds,
            }
            steps.append(row)

            status = "ADMIT" if loyo_ok else "reject (LOYO)"
            print(
                f"  Step {step_idx} {status}: {c} | F={comp['F']:.4f} df=({comp['df_num']},{comp['df_den']}) "
                f"p={comp['p_value']:.4g} dR2={comp['delta_r2']:.4f} | "
                f"LOYO RMSE red={rmse_r:.3f} full={rmse_f:.3f} (n_folds={n_folds})"
            )

            if loyo_ok:
                selected.append(c)
                round_admitted = True
                break
            loyo_failed.add(c)

        if not round_admitted:
            print(
                "  No candidate passed LOYO this round; stopping forward selection.",
                file=sys.stderr,
            )
            break

    return selected, steps, loyo_failed, present


def print_skill_selection_summary(
    df: pd.DataFrame,
    y_col: str,
    candidate_order: list[str],
    present: list[str],
    selected: list[str],
    loyo_failed: set[str],
    tournament_cols: list[str],
    alpha: float,
) -> None:
    """
    Print which skill metrics were used vs rejected, with reasons (stdout).
    End-state Gate A test for non-selected present candidates vs tournament + selected.
    """
    unavailable = [c for c in candidate_order if c not in present]
    rejected_loyo = sorted(c for c in loyo_failed if c in present)
    not_selected_present = [c for c in present if c not in selected]

    insufficient: list[str] = []
    not_sig: list[tuple[str, float, float, float]] = []  # name, p, F, dR2
    edge_sig: list[tuple[str, float, float, float]] = []

    cols_red_final = tournament_cols + selected
    for c in not_selected_present:
        if c in loyo_failed:
            continue
        cols_full = cols_red_final + [c]
        comp = nested_ols_compare(df, y_col, cols_red_final, cols_full)
        if comp is None:
            insufficient.append(c)
            continue
        if comp["p_value"] >= alpha:
            not_sig.append((c, comp["p_value"], comp["F"], comp["delta_r2"]))
        else:
            edge_sig.append((c, comp["p_value"], comp["F"], comp["delta_r2"]))

    print("\n=== Skill metrics summary (forward selection) ===")
    print("\nTournament block (always in model):")
    for col in tournament_cols:
        print(f"  - {col}")
    print(f"\nSkill metrics admitted ({len(selected)}):")
    if selected:
        for col in selected:
            print(f"  - {col}")
    else:
        print("  (none)")

    print(f"\nSkill metrics not admitted — unavailable in merged data ({len(unavailable)}):")
    if unavailable:
        for col in unavailable:
            print(f"  - {col}")
    else:
        print("  (none)")

    print(f"\nSkill metrics not admitted — passed Gate A but failed LOYO ({len(rejected_loyo)}):")
    if rejected_loyo:
        for col in rejected_loyo:
            print(f"  - {col}")
    else:
        print("  (none)")

    print(
        f"\nSkill metrics not admitted — not significant vs final model "
        f"(Gate A, p >= {alpha}; end-state nested OLS) ({len(not_sig)}):"
    )
    if not_sig:
        for col, p_v, f_v, dr in sorted(not_sig, key=lambda x: x[1]):
            print(f"  - {col}  (p={p_v:.4g}, F={f_v:.4f}, dR2={dr:.4f})")
    else:
        print("  (none)")

    print(f"\nSkill metrics — insufficient data for end-state test ({len(insufficient)}):")
    if insufficient:
        for col in insufficient:
            print(f"  - {col}")
    else:
        print("  (none)")

    print(
        f"\nSkill metrics — significant vs final model but not in model "
        f"(p < {alpha}; path/order; rare) ({len(edge_sig)}):"
    )
    if edge_sig:
        for col, p_v, f_v, dr in sorted(edge_sig, key=lambda x: x[1]):
            print(f"  - {col}  (p={p_v:.4g}, F={f_v:.4f}, dR2={dr:.4f})")
    else:
        print("  (none)")


def run_model_iteration(
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> tuple[RegressionResultsWrapper, list[str]]:
    if not feature_cols:
        raise ValueError("run_model_iteration: feature_cols is empty.")
    work = df.dropna(subset=["finish_rank"] + feature_cols).copy()
    if len(work) == 0:
        chk = ["finish_rank"] + feature_cols
        na = df[[c for c in chk if c in df.columns]].isna().sum()
        raise ValueError(
            "No complete cases for OLS (n=0). Missing-count per column:\n"
            f"{na.to_string()}"
        )
    y = work["finish_rank"].to_numpy(dtype=float)
    X = sm.add_constant(work[feature_cols], has_constant="add")

    groups = work["dg_id"].to_numpy() if "dg_id" in work.columns else None

    if groups is not None:
        try:
            res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        except Exception:
            # e.g. covariance failure when clusters are degenerate; fall back to classical SE
            res = sm.OLS(y, X).fit()
    else:
        res = sm.OLS(y, X).fit()

    print(f"\n=== OLS ({label}) n={len(work)} ===")
    print(res.summary())
    return res, feature_cols


def anova_block_test(
    df: pd.DataFrame,
    all_feats: list[str],
    block: list[str],
) -> tuple[float, float, int]:
    work = df.dropna(subset=["finish_rank"] + all_feats).copy()
    y = work["finish_rank"].to_numpy(dtype=float)
    keep = [c for c in all_feats if c not in block]
    Xf = work[all_feats]
    Xr = work[keep]
    return partial_f_test(y, Xf, Xr, add_const_full=True, add_const_reduced=True)


def impute_group_medians(df: pd.DataFrame, cols: list[str], group_col: str) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(float)
        df[c] = df.groupby(group_col)[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())


def fill_feature_columns_for_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "season_year",
) -> None:
    """
    In-place: coerce to float, impute by group then global median so listwise-complete
    regression uses the same n as nested comparisons that dropna on subsets of columns.

    Forward selection fits tournament + a growing skill set; the final model uses all
    selected skills at once — without this pass, disjoint NaNs across skills can leave
    zero complete cases.
    """
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column {c!r} is missing from training frame.")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    impute_group_medians(df, feature_cols, group_col)
    for c in feature_cols:
        col_med = df[c].median()
        fill_v = 0.0 if pd.isna(col_med) else float(col_med)
        df[c] = df[c].fillna(fill_v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Masters regression (6m PGA + prior Masters)")
    ap.add_argument(
        "--require-top25-in-6m",
        action="store_true",
        help="Keep only players with at least one top-25 in the prior-6-month PGA window",
    )
    ap.add_argument(
        "--include-skill-features",
        action="store_true",
        help="Append fixed skill bundle (no forward selection)",
    )
    ap.add_argument(
        "--skill-forward-select",
        action="store_true",
        help="Iteratively add skill metrics if p < alpha and LOYO RMSE improves",
    )
    ap.add_argument(
        "--entry-alpha",
        type=float,
        default=0.05,
        help="Gate A: maximum p-value for adding a skill candidate (default 0.05)",
    )
    ap.add_argument(
        "--loyo-epsilon",
        type=float,
        default=0.0,
        help="Gate B: admit if mean LOYO RMSE(full) <= mean LOYO RMSE(reduced) - epsilon",
    )
    ap.add_argument(
        "--masters-decay-lambda",
        type=float,
        default=0.25,
        help="Exponential decay on prior Masters finishes: weight ∝ exp(-λ * years_before_Masters)",
    )
    ap.add_argument(
        "--masters-recent-best-years",
        type=int,
        default=8,
        help="Best prior Masters finish uses only rounds in this many calendar years before target",
    )
    ap.add_argument(
        "--min-pga6m-starts",
        type=int,
        default=3,
        help="Below this many PGA starts in the 6m window, form metrics are pessimistic + flag",
    )
    ap.add_argument(
        "--pga-ref-year",
        type=int,
        default=2026,
        help="Reference year for back-casting skill-table age on training rows (age_at_Masters)",
    )
    ap.add_argument(
        "--no-skip-predict-below-min-pga6m",
        action="store_true",
        help="Include sub-threshold 6m-start players in the ranked 2026 prediction table",
    )
    args = ap.parse_args()

    if args.include_skill_features and args.skill_forward_select:
        print(
            "Both --include-skill-features and --skill-forward-select set; "
            "using forward selection only.",
            file=sys.stderr,
        )

    key = os.environ.get("DATAGOLF_KEY", "").strip()
    if not key:
        print("Set DATAGOLF_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    fc = FetchConfig(key=key)

    print("Fetching PGA event list (historical-raw-data/event-list)…")
    event_list = fetch_pga_event_list(fc)
    masters_meta = masters_rows_from_event_list(event_list)
    train_years = {y for y, _ in MASTERS_ARCHIVE}
    meta_years = train_years | {2026}
    masters_meta = masters_meta[masters_meta["calendar_year"].isin(meta_years)]
    print(f"  Using {len(masters_meta)} Masters rows for years {sorted(meta_years)}.")

    extra_2026 = fetch_schedule_masters_2026(fc)
    if extra_2026:
        print(f"  2026 Masters from schedule: event_id={extra_2026[0]} date={extra_2026[1]}")

    print("Fetching pre-tournament archives for 6-month windows + Masters history…")
    archive_cache = collect_needed_archives(fc, event_list, masters_meta, extra_2026)

    mhist_long = build_masters_history_long(fc, archive_cache)
    print(f"  Masters history long table: {len(mhist_long)} player-years.")

    frames: list[pd.DataFrame] = []
    for year, eid in MASTERS_ARCHIVE:
        df_y = fetch_archive_year(fc, year, eid)
        frames.append(df_y)
        print(f"  Masters archive label year={year} event_id={eid}: {len(df_y)} rows")

    train = pd.concat(frames, ignore_index=True)
    train = train.dropna(subset=["finish_rank"])

    form_parts = []
    prior_parts = []
    for cy, _eid in MASTERS_ARCHIVE:
        fdf = form_features_for_year(event_list, masters_meta, cy, archive_cache)
        if fdf.empty:
            continue
        fdf["season_year"] = cy
        form_parts.append(fdf)

        pm = prior_masters_features_decay(
            mhist_long,
            cy,
            args.masters_decay_lambda,
            args.masters_recent_best_years,
        )
        pm["season_year"] = cy
        prior_parts.append(pm)

    form_all = pd.concat(form_parts, ignore_index=True) if form_parts else pd.DataFrame()
    prior_all = pd.concat(prior_parts, ignore_index=True) if prior_parts else pd.DataFrame()

    train = train.merge(form_all, on=["dg_id", "season_year"], how="left")
    train = train.merge(prior_all, on=["dg_id", "season_year"], how="left")

    print("Fetching skill / approach / decomp tables (age + optional skill merge)…")
    skill_tbl, approach_tbl, decomp_tbl = fetch_skill_tables(fc)
    train = merge_age_at_masters(train, skill_tbl, args.pga_ref_year)

    prior_cols = [
        "masters_prior_starts",
        "masters_prior_decay_mean_finish",
        "masters_prior_best_recent",
    ]
    impute_group_medians(train, prior_cols, "season_year")

    form_cols = [
        "pga6m_starts",
        "pga6m_mean_finish",
        "pga6m_top25_rate",
        "pga6m_made_cut_rate",
    ]
    impute_group_medians(train, form_cols, "season_year")
    apply_pga6m_insufficient_form(train, args.min_pga6m_starts)
    impute_group_medians(train, ["age_at_masters"], "season_year")

    if args.require_top25_in_6m:
        train = train[train["pga6m_top25_count"] >= 1].copy()
        print(f"\nAfter --require-top25-in-6m: {len(train)} rows")

    tournament_features = prior_cols + form_cols + ["pga6m_insufficient", "age_at_masters"]

    skill_bundle = [
        "sg_t2g",
        "sg_app",
        "par5_long_approach_proxy",
        "course_history",
        "short_game_tight_lie_proxy",
        "length_proxy",
    ]

    selected_skills: list[str] = []
    if args.skill_forward_select:
        train = merge_skill_features(train, skill_tbl, approach_tbl, decomp_tbl)
        skill_cols_avail = [c for c in SKILL_CANDIDATE_ORDER if c in train.columns]
        for c in skill_cols_avail:
            train[c] = pd.to_numeric(train[c], errors="coerce")
        impute_group_medians(train, skill_cols_avail, "season_year")
        for c in skill_cols_avail:
            train[c] = train[c].fillna(train[c].median())
        skill_cols_avail = [c for c in skill_cols_avail if not train[c].isna().all()]
        train = train.dropna(subset=tournament_features)
        selected_skills, _steps, loyo_failed_set, present_candidates = forward_select_skills(
            train,
            "finish_rank",
            tournament_features,
            SKILL_CANDIDATE_ORDER,
            alpha=args.entry_alpha,
            loyo_epsilon=args.loyo_epsilon,
            group_col="season_year",
        )
        feature_cols = tournament_features + selected_skills
        print_skill_selection_summary(
            train,
            "finish_rank",
            SKILL_CANDIDATE_ORDER,
            present_candidates,
            selected_skills,
            loyo_failed_set,
            tournament_features,
            args.entry_alpha,
        )
    elif args.include_skill_features:
        train = merge_skill_features(train, skill_tbl, approach_tbl, decomp_tbl)
        feature_cols = tournament_features + skill_bundle
    else:
        feature_cols = list(tournament_features)

    fill_feature_columns_for_regression(train, feature_cols, group_col="season_year")
    train = train.dropna(subset=feature_cols)
    if len(train) == 0:
        raise SystemExit(
            "No training rows after feature assembly and imputation; check merges and filters."
        )

    print(
        "\nFeatures: tournament block + "
        f"{'forward-selected skills' if args.skill_forward_select else 'optional skill bundle' if args.include_skill_features else 'skills off'}."
    )

    res_full, _ = run_model_iteration(
        train,
        feature_cols,
        "final (cluster SE by dg_id if available)",
    )
    model_for_predict = res_full
    predict_features = list(feature_cols)

    blocks: dict[str, list[str]] = {
        "prior_masters": [c for c in prior_cols if c in feature_cols],
        "pga_last_6m": [
            c for c in form_cols + ["pga6m_insufficient"] if c in feature_cols
        ],
        "age": [c for c in ["age_at_masters"] if c in feature_cols],
    }
    if selected_skills:
        blocks["skill_selected"] = selected_skills
    elif args.include_skill_features and not args.skill_forward_select:
        blocks["skill_extra"] = [c for c in skill_bundle if c in feature_cols]

    print(
        "\n=== Partial F-tests (nested OLS; classical F, not cluster-robust) ===\n"
        "Final fit above uses cluster-robust SE where possible; these F-tests match "
        "the forward-selection Gate A logic.",
    )
    block_results: list[tuple[str, float, float, int]] = []
    for name, cols in blocks.items():
        cols = [c for c in cols if c in feature_cols]
        if len(cols) == 0:
            continue
        f_s, p_v, q = anova_block_test(train, feature_cols, cols)
        block_results.append((name, f_s, p_v, q))
        print(f"  drop {name} ({cols}): F={f_s:.3f}, df_num={q}, p={p_v:.4g}")

    insignificant = [b for b in block_results if b[2] > 0.05]
    if (
        not args.skill_forward_select
        and not args.include_skill_features
        and len(insignificant) == len(block_results)
        and len(block_results) > 0
    ):
        print(
            "\nAll tested tournament blocks insignificant at α=0.05 — refitting without "
            "`pga6m_mean_finish`."
        )
        reduced = [c for c in feature_cols if c != "pga6m_mean_finish"]
        res_reduced, _ = run_model_iteration(train, reduced, "reduced (dropped pga6m_mean_finish)")
        model_for_predict = res_reduced
        predict_features = reduced
    elif any(b[2] <= 0.05 for b in block_results):
        print("\nSome omitted blocks are significant at α=0.05.")
        for name, _, p_v, _ in [b for b in block_results if b[2] <= 0.05]:
            print(f"  - `{name}` p={p_v:.4g}")

    # --- 2026 predictions ---
    print("\n=== 2026 Masters projected field ===")
    if not extra_2026:
        print("Could not resolve 2026 Masters date from schedule; skipping predictions.")
        return

    eid26, d26 = extra_2026
    w0, w1 = six_month_window_before(d26)
    evs26 = pga_events_in_window(event_list, w0, w1)
    dfs26 = [archive_cache.get((int(e["event_id"]), int(e["calendar_year"]))) for e in evs26]
    form26 = aggregate_event_finishes(dfs26)

    prior26 = prior_masters_features_decay(
        mhist_long,
        2026,
        args.masters_decay_lambda,
        args.masters_recent_best_years,
    )

    field = fetch_field(fc)
    live = fetch_live_pre_tournament(fc)
    merged = field.merge(live[["dg_id", "top_20", "win"]], on="dg_id", how="inner")
    merged = merged.merge(form26, on="dg_id", how="left")
    merged = merged.merge(prior26, on="dg_id", how="left")
    merged = merge_age_at_masters(merged, skill_tbl, args.pga_ref_year, use_direct_age=True)

    for c in prior_cols:
        if c in merged.columns:
            merged[c] = merged[c].astype(float)
    merged[prior_cols] = merged[prior_cols].fillna(merged[prior_cols].median())

    for c in form_cols:
        if c in merged.columns:
            merged[c] = merged[c].astype(float)
    suff_mask = (
        pd.to_numeric(merged["pga6m_starts"], errors="coerce").fillna(0.0)
        >= float(args.min_pga6m_starts)
    )
    med_form = merged.loc[suff_mask, form_cols].median()
    if not bool(suff_mask.any()):
        med_form = train[form_cols].median()
    for c in form_cols:
        if c in merged.columns:
            merged.loc[suff_mask, c] = merged.loc[suff_mask, c].fillna(med_form.get(c, np.nan))
    apply_pga6m_insufficient_form(merged, args.min_pga6m_starts)

    age_med_tr = train["age_at_masters"].median()
    merged["age_at_masters"] = pd.to_numeric(merged["age_at_masters"], errors="coerce").fillna(
        age_med_tr
    )

    if args.require_top25_in_6m:
        merged = merged[merged["pga6m_top25_count"] >= 1].copy()

    if args.skill_forward_select or args.include_skill_features:
        merged = merge_skill_features(merged, skill_tbl, approach_tbl, decomp_tbl)
        if args.skill_forward_select:
            for c in predict_features:
                if c not in tournament_features and c in merged.columns:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce")
            # Impute skill columns for prediction rows
            skill_only_pred = [c for c in predict_features if c not in tournament_features]
            if skill_only_pred:
                merged[skill_only_pred] = merged[skill_only_pred].fillna(
                    merged[skill_only_pred].median()
                )

    if not args.no_skip_predict_below_min_pga6m:
        starts_pred = pd.to_numeric(merged["pga6m_starts"], errors="coerce").fillna(0.0)
        low_starts = starts_pred < float(args.min_pga6m_starts)
        if low_starts.any():
            n_ex = int(low_starts.sum())
            name_col = "player_name" if "player_name" in merged.columns else "dg_id"
            names = merged.loc[low_starts, name_col].astype(str).tolist()
            head = ", ".join(names[:40])
            tail = " …" if len(names) > 40 else ""
            print(
                f"\nExcluding {n_ex} player(s) from ranked 2026 predictions "
                f"(pga6m_starts < {args.min_pga6m_starts}). "
                "Use --no-skip-predict-below-min-pga6m to include them."
            )
            print(f"  {head}{tail}")
        merged = merged.loc[~low_starts].copy()

    pred_df = merged.dropna(subset=predict_features).copy()
    Xp = sm.add_constant(pred_df[predict_features], has_constant="add")
    pred_df["pred_finish_rank"] = model_for_predict.predict(Xp)

    pred_df = pred_df.sort_values("pred_finish_rank")
    cols_show = ["player_name", "pred_finish_rank"] + predict_features
    cols_show = [c for c in cols_show if c in pred_df.columns]
    print(pred_df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
