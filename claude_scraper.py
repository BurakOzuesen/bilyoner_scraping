#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bilyoner odds & score collector — optimized

Key improvements:
- Connection pooling + robust retries (requests.Session + urllib3 Retry)
- Parallel HTTP for odds & scores (ThreadPoolExecutor) with bounded workers
- Clean schema handling (nullable Int64 for scores), no iterrows loops
- Atomic CSV writes, idempotent appends, column reindexing
- Timeouts everywhere, explicit error logging, graceful backoff
- CLI options for flexibility
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import tempfile
import argparse
from typing import Any, Dict, List, Optional, Tuple
from random import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ------------------------------- Config -------------------------------- #

CSV_PATH = "data/processed/match_odds_cleaned_20250801.csv"
KEY_COLS = [
    "match_date", "match_time", "tournament", "match_id", "homeTeam", "awayTeam",
    "firstHalfHomeGoal", "firstHalfAwayGoal", "totalHomeGoal", "totalAwayGoal",
    "homeCorner", "awayCorner",
]

USER_AGENT = "Mozilla/5.0 (compatible; OddsCollector/1.0; +https://example.local)"
BASE = "https://www.bilyoner.com"

DEFAULT_TIMEOUT = (5, 15)  # (connect, read) seconds
MAX_WORKERS = max(4, os.cpu_count() or 4)

ODD_FILTER_EXCLUDES = ("oyuncu", "kart", "korner", "penaltı", "özel", "dakikalar", "nasıl", "aralığ")

# ------------------------------- HTTP ---------------------------------- #

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

def _get(url: str, *, params: Optional[dict] = None, timeout=DEFAULT_TIMEOUT) -> Optional[dict]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        return None
    return None

# ------------------------------- Core ---------------------------------- #

def get_match_ids(shuffle_ids: bool = True) -> List[str]:
    """
    Günlük maç bülteninden match_id listesi döner.
    """
    url = f"{BASE}/api/v3/mobile/aggregator/gamelist/all/v1"
    params = {"tabType": 1, "bulletinType": 2, "liveEventsEnabledForPreBulletin": "true"}
    data = _get(url, params=params)
    events = (data or {}).get("events", {})
    ids = list(events.keys())
    if shuffle_ids:
        shuffle(ids)
    return [str(x) for x in ids]

def _extract_odds_rows(odds_json: dict) -> Dict[str, Any]:
    """
    Pulls columns from 'Tümü' tab and filters irrelevant markets.
    Returns a dict of { "Market :: Selection": odd_value }.
    """
    out: Dict[str, Any] = {}
    for tab in (odds_json or {}).get("oddGroupTabs", []):
        if tab.get("title") != "Tümü":
            continue
        for market in tab.get("matchCardOdds", []):
            mname = (market.get("name") or "").strip()
            if any(ex in mname.lower() for ex in ODD_FILTER_EXCLUDES):
                continue
            for odd in market.get("oddList", []):
                name, val = odd.get("n"), odd.get("val")
                if name is None or val is None:
                    continue
                out[f"{mname} :: {name}"] = val
    return out

def fetch_match_odds(match_id: str, *, is_live: bool = True, is_popular: bool = False) -> Optional[Dict[str, Any]]:
    info_url = f"{BASE}/api/v3/mobile/aggregator/gamelist/events/{match_id}"
    odds_url = f"{BASE}/api/v3/mobile/aggregator/match-card/{match_id}/odds"
    params = {"isLiveEvent": str(is_live).lower(), "isPopular": str(is_popular).lower()}

    info = _get(info_url)
    odds = _get(odds_url, params=params)
    if not info or not odds:
        return None

    odds_dict = _extract_odds_rows(odds)
    if not odds_dict:
        return None

    esd: str = info.get("esd") or ""
    date, time_str = (esd.split("T") + [None])[:2]

    row: Dict[str, Any] = {
        **odds_dict,
        "match_date": date,
        "match_time": time_str,
        "tournament": info.get("lgn"),
        "match_id": str(match_id),
        "homeTeam": odds.get("homeTeam"),
        "awayTeam": odds.get("awayTeam"),
        "firstHalfHomeGoal": pd.NA,
        "firstHalfAwayGoal": pd.NA,
        "totalHomeGoal": pd.NA,
        "totalAwayGoal": pd.NA,
        "homeCorner": pd.NA,
        "awayCorner": pd.NA,
    }
    return row

def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), suffix=".csv") as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)

def _read_master(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"match_id": "string"})
    else:
        df = pd.DataFrame(columns=KEY_COLS)
        df["match_id"] = df["match_id"].astype("string")
    # enforce nullable integer for score columns
    for c in ["firstHalfHomeGoal", "firstHalfAwayGoal", "totalHomeGoal", "totalAwayGoal", "homeCorner", "awayCorner"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
        else:
            df[c] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df

def append_matches_to_csv(match_ids: List[str], csv_path: str = CSV_PATH, *, max_workers: int = MAX_WORKERS) -> None:
    master = _read_master(csv_path)
    existing = set(master["match_id"].astype("string").dropna().tolist())

    new_ids = [mid for mid in map(str, match_ids) if mid not in existing]
    if not new_ids:
        print("ℹ️ Eklenebilecek yeni maç bulunamadı.")
        return

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_match_odds, mid): mid for mid in new_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Odds fetch"):
            mid = futures[fut]
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                # swallow to keep going
                continue

    if not rows:
        print("ℹ️ Yeni veriler alınamadı.")
        return

    new_df = pd.DataFrame(rows)
    # unify columns (schema evolution)
    all_cols = list(dict.fromkeys(KEY_COLS + sorted([c for c in new_df.columns if c not in KEY_COLS])))
    master = master.reindex(columns=list(dict.fromkeys(master.columns.tolist() + all_cols)))
    new_df = new_df.reindex(columns=master.columns)

    combined = pd.concat([master, new_df], ignore_index=True)
    # ensure dtypes for IDs/scores
    combined["match_id"] = combined["match_id"].astype("string")
    for c in ["firstHalfHomeGoal", "firstHalfAwayGoal", "totalHomeGoal", "totalAwayGoal", "homeCorner", "awayCorner"]:
        if c in combined.columns:
            combined[c] = combined[c].astype("Int64")

    _atomic_write_csv(combined, csv_path)
    print(f"✅ {len(new_df)} yeni maç eklendi → {csv_path}")

def _parse_scores(score_json: dict) -> Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
    if not score_json:
        return None
    scores = {s.get("scoreName"): s for s in (score_json.get("score") or [])}
    try:
        fh = scores.get("SOCCER_FIRST_HALF")
        ft = scores.get("SOCCER_END_SCORE")
        fh_h = int(fh["homeScore"]) if fh and fh.get("homeScore") is not None else None
        fh_a = int(fh["awayScore"]) if fh and fh.get("awayScore") is not None else None
        ft_h = int(ft["homeScore"]) if ft and ft.get("homeScore") is not None else None
        ft_a = int(ft["awayScore"]) if ft and ft.get("awayScore") is not None else None
        return fh_h, fh_a, ft_h, ft_a
    except Exception:
        return None

def _fetch_score_for_id(match_id: str) -> Optional[Tuple[str, Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]]:
    url = f"{BASE}/api/mobile/match-card/v2/{match_id}/status"
    # explicit param per original code: sgSportTypeId=1 (soccer)
    data = _get(url, params={"sgSportTypeId": 1})
    parsed = _parse_scores(data or {})
    if parsed is None:
        return None
    return str(match_id), parsed

def update_scores_in_csv(csv_path: str = CSV_PATH, *, max_workers: int = MAX_WORKERS) -> None:
    df = _read_master(csv_path)
    # rows missing either first-half pair or full-time pair
    need_mask = df["firstHalfHomeGoal"].isna() | df["firstHalfAwayGoal"].isna() | df["totalHomeGoal"].isna() | df["totalAwayGoal"].isna()
    todo = df.loc[need_mask, "match_id"].astype("string").dropna().unique().tolist()
    if not todo:
        print("ℹ️ Güncellenecek skor yok.")
        return

    results: Dict[str, Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_score_for_id, mid): mid for mid in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Score fetch"):
            try:
                r = fut.result()
                if r:
                    mid, vals = r
                    results[mid] = vals
            except Exception:
                continue

    if not results:
        print("ℹ️ Skor güncellenemedi.")
        return

    # Map back without loops
    ser_fh_h = pd.Series({k: v[0] for k, v in results.items()}, name="firstHalfHomeGoal", dtype="Int64")
    ser_fh_a = pd.Series({k: v[1] for k, v in results.items()}, name="firstHalfAwayGoal", dtype="Int64")
    ser_ft_h = pd.Series({k: v[2] for k, v in results.items()}, name="totalHomeGoal", dtype="Int64")
    ser_ft_a = pd.Series({k: v[3] for k, v in results.items()}, name="totalAwayGoal", dtype="Int64")

    df = df.set_index("match_id")
    for name, ser in [("firstHalfHomeGoal", ser_fh_h), ("firstHalfAwayGoal", ser_fh_a),
                      ("totalHomeGoal", ser_ft_h), ("totalAwayGoal", ser_ft_a)]:
        # only fill missing to avoid overwriting existing values
        missing = df[name].isna()
        df.loc[missing, name] = ser.reindex(df.index)[missing]

    df = df.reset_index()
    _atomic_write_csv(df, csv_path)
    print("✅ Skorlar güncellendi ve CSV’ye yazıldı.")

# ------------------------------- CLI ----------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Bilyoner odds & scores to CSV.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_all = sub.add_parser("run", help="Fetch odds then update scores (default).")
    p_all.add_argument("--csv", default=CSV_PATH)
    p_all.add_argument("--workers", type=int, default=MAX_WORKERS)
    p_all.add_argument("--no-shuffle", action="store_true")

    p_odds = sub.add_parser("odds", help="Only fetch & append new odds.")
    p_odds.add_argument("--csv", default=CSV_PATH)
    p_odds.add_argument("--workers", type=int, default=MAX_WORKERS)
    p_odds.add_argument("--no-shuffle", action="store_true")

    p_sc = sub.add_parser("scores", help="Only update scores.")
    p_sc.add_argument("--csv", default=CSV_PATH)
    p_sc.add_argument("--workers", type=int, default=MAX_WORKERS)

    args = parser.parse_args(argv)

    cmd = args.cmd or "run"
    if cmd in ("run", "odds"):
        ids = get_match_ids(shuffle_ids=not getattr(args, "no_shuffle", False))
        append_matches_to_csv(ids, csv_path=args.csv, max_workers=args.workers)

    if cmd in ("run", "scores"):
        update_scores_in_csv(csv_path=getattr(args, "csv", CSV_PATH), max_workers=getattr(args, "workers", MAX_WORKERS))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
