import pandas as pd
import requests
import os
import math
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

CSV_PATH = "data/processed/match_odds_cleaned_20250801.csv"
KEY_COLS = ["match_date", "match_time", "tournament", "match_id", "homeTeam", "awayTeam",
            "firstHalfHomeGoal", "firstHalfAwayGoal", "totalHomeGoal", "totalAwayGoal",
            "homeCorner", "awayCorner"]

headers = {"User-Agent": "Mozilla/5.0"}

def get_match_ids():
    """
    Günlük maç bülteninden match_id listesi döner.
    """
    url = "https://www.bilyoner.com/api/v3/mobile/aggregator/gamelist/all/v1"
    params = {"tabType": 1, "bulletinType": 2, "liveEventsEnabledForPreBulletin": "true"}

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    match_ids = list(data.get("events", {}).keys())
    return match_ids

def fetch_match_odds(match_id, is_live=True, is_popular=False):
    """
    Tek maç için tüm geçerli oranları ve temel bilgileri döner.
    """
    info_url = f"https://www.bilyoner.com/api/v3/mobile/aggregator/gamelist/events/{match_id}"
    odds_url = f"https://www.bilyoner.com/api/v3/mobile/aggregator/match-card/{match_id}/odds"
    params = {"isLiveEvent": str(is_live).lower(), "isPopular": str(is_popular).lower()}

    try:
        info = requests.get(info_url, headers=headers).json()
        odds = requests.get(odds_url, params=params, headers=headers).json()
    except:
        return None

    rows = []
    for item in odds.get("oddGroupTabs", []):
        if item["title"] != "Tümü":
            continue
        for market in item.get("matchCardOdds", []):
            mkt_name = market.get("name", "")
            if any(ex in mkt_name.lower() for ex in ["oyuncu", "kart", "korner", "penaltı", "özel", "dakikalar", "nasıl", "aralığ"]):
                continue
            for odd in market.get("oddList", []):
                name = odd.get("n")
                value = odd.get("val")
                if name is None or value is None:
                    continue
                colname = f"{mkt_name} :: {name}"
                rows.append((colname, value))

    if not rows:
        return None

    row_dict = {k: v for k, v in rows}
    row_dict.update({
        "match_date": info.get("esd", "").split("T")[0],
        "match_time": info.get("esd", "").split("T")[1] if "T" in info.get("esd", "") else None,
        "tournament": info.get("lgn"),
        "match_id": match_id,
        "homeTeam": odds.get("homeTeam"),
        "awayTeam": odds.get("awayTeam"),
        "firstHalfHomeGoal": None,
        "firstHalfAwayGoal": None,
        "totalHomeGoal": None,
        "totalAwayGoal": None,
        "homeCorner": None,
        "awayCorner": None
    })

    return pd.DataFrame([row_dict])

def append_matches_to_csv(match_ids, csv_path=CSV_PATH):
    """
    Yeni match_id'leri çekip CSV’ye ekler. 
    Eğer match_id zaten varsa tekrar eklenmez.
    """
    if os.path.exists(csv_path):
        master_df = pd.read_csv(csv_path, dtype={"match_id": str})
    else:
        master_df = pd.DataFrame(columns=KEY_COLS)

    existing_ids = set(master_df["match_id"].astype(str)) if not master_df.empty else set()

    new_rows = []

    for mid in tqdm(match_ids):
        mid = str(mid)

        # Eğer zaten varsa atla
        if mid in existing_ids:
            continue

        df = fetch_match_odds(mid)
        if df is None:
            continue

        # Kolon tamamlama
        for col in df.columns:
            if col not in master_df.columns:
                master_df[col] = pd.NA
        for col in master_df.columns:
            if col not in df.columns:
                df[col] = pd.NA

        new_rows.append(df)

    if new_rows:
        # Yeni maçları ekle
        new_data = pd.concat(new_rows, ignore_index=True)
        master_df = pd.concat([master_df, new_data], ignore_index=True)

        # Kolon sıralama
        other_cols = [c for c in master_df.columns if c not in KEY_COLS]
        master_df = master_df[KEY_COLS + sorted(other_cols)]

        master_df.to_csv(csv_path, index=False)
        print(f"\n✅ {len(new_rows)} yeni maç eklendi → {csv_path}")
    else:
        print("\nℹ️ Eklenebilecek yeni maç bulunamadı.")

def update_scores_in_csv(csv_path=CSV_PATH):
    """
    CSV'deki eksik skorları Bilyoner'den çekerek doldurur.
    """
    df = pd.read_csv(csv_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row["firstHalfHomeGoal"]) and pd.isna(row["firstHalfAwayGoal"]):
            match_id = row["match_id"]
            score_url = f"https://www.bilyoner.com/api/mobile/match-card/v2/{match_id}/status?sgSportTypeId=1"

            try:
                data = requests.get(score_url, headers=headers, timeout=10).json()
                scores = {s["scoreName"]: s for s in data.get("score", [])}
                df.at[idx, "firstHalfHomeGoal"] = int(scores["SOCCER_FIRST_HALF"]["homeScore"])
                df.at[idx, "firstHalfAwayGoal"] = int(scores["SOCCER_FIRST_HALF"]["awayScore"])
                df.at[idx, "totalHomeGoal"] = int(scores["SOCCER_END_SCORE"]["homeScore"])
                df.at[idx, "totalAwayGoal"] = int(scores["SOCCER_END_SCORE"]["awayScore"])
            except:
                continue

    df.to_csv(csv_path, index=False)
    print("✅ Skorlar güncellendi ve CSV’ye yazıldı.")

if __name__ == "__main__":
    ids = get_match_ids()
    append_matches_to_csv(ids)
    # update_scores_in_csv()
