# -*- coding: utf-8 -*-
# Fast value-bet finder (vectorized, modular)
# ------------------------------------------------------------
# Usage: python fast_value_bets.py
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime

# -------- Params --------
CSV_PATH = "match_odds_cleaned_20250801.csv"
DATA_DIR = "data/processed"
TOP_K = 100
OUT_PREFIX = "07092025_gpt"  # change if you like
SHUFFLE_SEED = 42        # deterministic
PRINT_SAMPLE = False      # turn on for quick sanity prints

KEY_COLS = [
    "match_date", "match_time", "tournament", "match_id",
    "homeTeam", "awayTeam",
    "firstHalfHomeGoal", "firstHalfAwayGoal",
    "totalHomeGoal", "totalAwayGoal",
    "homeCorner", "awayCorner"
]

# -------- IO --------
df = pd.read_csv(f"{DATA_DIR}/{CSV_PATH}")
df = df.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)

# Feature columns: numeric, non-key
numeric_cols = [c for c in df.columns if c not in KEY_COLS and pd.api.types.is_numeric_dtype(df[c])]

# Train (has final score), Test (missing final score)
train_df = df[df["totalHomeGoal"].notna()].reset_index(drop=True)
test_df  = df[df["totalHomeGoal"].isna()].reset_index(drop=True)

if len(test_df) == 0:
    raise RuntimeError("Test set is empty. No rows with missing totalHomeGoal/totalAwayGoal.")

# Pre-extract numeric matrices (float64) and NaN masks
T = train_df[numeric_cols].to_numpy(dtype=float, copy=True)   # (N_train, D)
X = test_df[numeric_cols].to_numpy(dtype=float, copy=True)    # (N_test,  D)
T_nan = np.isnan(T)
X_nan = np.isnan(X)

# Fast nan-aware Euclidean distance: per test row vs all train
def nan_euclidean_to_train(test_vec: np.ndarray, test_nan: np.ndarray) -> np.ndarray:
    """
    Returns distances to all rows in T.
    If no common valid features with a train row, distance = +inf.
    """
    # valid mask per train row: (~T_nan & ~test_nan) across features
    valid = (~T_nan) & (~test_nan)  # (N_train, D)

    # diff only where valid; elsewhere 0 so it doesn't contribute to sum
    diff = np.where(valid, T - test_vec, 0.0)
    # sums of squares over valid dims
    ss = np.einsum("ij,ij->i", diff, diff)
    # count valid dims; if 0 → inf distance
    cnt = valid.sum(axis=1)
    dist = np.sqrt(ss)
    dist[cnt == 0] = np.inf
    return dist

# Utility to safely pull odds from test row
def odd(test_row: pd.Series, label: str, default: float = 0.0) -> float:
    # some labels may not exist for some matches; fallback to 0
    return float(test_row.get(label, default) or 0.0)

# Helper to append a bet row if EV>1
def push_if_value(rows: list, test_row: pd.Series, label: str, count: int, odds: float):
    # count is 0..TOP_K → as percentage
    prob_pct = float(count)  # already percentage because TOP_K=100
    ev = (prob_pct * odds) / 100.0
    if ev > 1.0:
        rows.append({
            "match_date": test_row["match_date"],
            "match_time": test_row["match_time"],
            "match_id":   test_row["match_id"],
            "hometeam":   test_row["homeTeam"],
            "awayteam":   test_row["awayTeam"],
            "bet_name":   label,
            "probability": prob_pct,
            "odds":        odds
        })

# Build all bet counts from the top-K neighbors (vectorized on arrays)
def compute_bet_counts_and_push(rows: list, test_row: pd.Series, neigh: pd.DataFrame):
    # short aliases as NumPy arrays
    hg  = neigh["totalHomeGoal"].to_numpy()
    ag  = neigh["totalAwayGoal"].to_numpy()
    ihg = neigh["firstHalfHomeGoal"].to_numpy()
    iag = neigh["firstHalfAwayGoal"].to_numpy()

    shg = hg - ihg
    sag = ag - iag

    # frequently reused composites
    tg  = hg + ag
    itg = ihg + iag

    # --- Example: MS 1 / X / 2 ---
    ms1 = (hg > ag)
    msx = (hg == ag)
    ms2 = (ag > hg)
    push_if_value(rows, test_row, "Maç Sonucu :: MS 1", ms1.sum(), odd(test_row, "Maç Sonucu :: MS 1"))
    push_if_value(rows, test_row, "Maç Sonucu :: MS X", msx.sum(), odd(test_row, "Maç Sonucu :: MS X"))
    push_if_value(rows, test_row, "Maç Sonucu :: MS 2", ms2.sum(), odd(test_row, "Maç Sonucu :: MS 2"))

    # --- Deplasman Gol Alt/Üst ---
    dep_thr = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5]
    dep_lbl = ["0,5","1,5","2,5","3,5","4,5","6,5"]
    for v, s in zip(dep_thr, dep_lbl):
        push_if_value(rows, test_row, f"Deplasman Gol Alt/Üst :: Dep {s} Alt", (ag < v).sum(), odd(test_row, f"Deplasman Gol Alt/Üst :: Dep {s} Alt"))
        push_if_value(rows, test_row, f"Deplasman Gol Alt/Üst :: Dep {s} Üst", (ag >= v).sum(), odd(test_row, f"Deplasman Gol Alt/Üst :: Dep {s} Üst"))

    # Deplasman gol yemeden kazanır mı?
    push_if_value(rows, test_row, "Deplasman Gol Yemeden Kazanır mı? :: Evet", ((ag > hg) & (hg == 0)).sum(), odd(test_row, "Deplasman Gol Yemeden Kazanır mı? :: Evet"))
    push_if_value(rows, test_row, "Deplasman Gol Yemeden Kazanır mı? :: Hayır", (~((ag > hg) & (hg == 0))).sum(), odd(test_row, "Deplasman Gol Yemeden Kazanır mı? :: Hayır"))

    # Deplasman hangi yarıda daha fazla?
    push_if_value(rows, test_row, "Deplasman Hangi Yarıda Daha Fazla Gol Atar? :: 1. Yarı", (iag > sag).sum(), odd(test_row, "Deplasman Hangi Yarıda Daha Fazla Gol Atar? :: 1. Yarı"))
    push_if_value(rows, test_row, "Deplasman Hangi Yarıda Daha Fazla Gol Atar? :: 2. Yarı", (sag > iag).sum(), odd(test_row, "Deplasman Hangi Yarıda Daha Fazla Gol Atar? :: 2. Yarı"))
    push_if_value(rows, test_row, "Deplasman Hangi Yarıda Daha Fazla Gol Atar? :: Eşit", (iag == sag).sum(), odd(test_row, "Deplasman Hangi Yarıda Daha Fazla Gol Atar? :: Eşit"))

    # Deplasman her iki yarıyı da kazanır mı?
    push_if_value(rows, test_row, "Deplasman Her İki Yarıyı da Kazanır mı? :: Evet",
                  ((iag > ihg) & (sag > shg)).sum(),
                  odd(test_row, "Deplasman Her İki Yarıyı da Kazanır mı? :: Evet"))
    push_if_value(rows, test_row, "Deplasman Her İki Yarıyı da Kazanır mı? :: Hayır",
                  (~((iag > ihg) & (sag > shg))).sum(),
                  odd(test_row, "Deplasman Her İki Yarıyı da Kazanır mı? :: Hayır"))

    # Deplasman herhangi bir yarıyı kazanır mı?
    push_if_value(rows, test_row, "Deplasman Herhangi Bir Yarıyı Kazanır :: Evet",
                  ((iag > ihg) | (sag > shg)).sum(),
                  odd(test_row, "Deplasman Herhangi Bir Yarıyı Kazanır :: Evet"))
    push_if_value(rows, test_row, "Deplasman Herhangi Bir Yarıyı Kazanır :: Hayır",
                  (~((iag > ihg) | (sag > shg))).sum(),
                  odd(test_row, "Deplasman Herhangi Bir Yarıyı Kazanır :: Hayır"))

    # Deplasman İY alt/üst
    dep_iy_thr = [0.5, 1.5, 2.5]
    dep_iy_lbl = ["0,5","1,5","2,5"]
    for v, s in zip(dep_iy_thr, dep_iy_lbl):
        push_if_value(rows, test_row, f"Deplasman İlk Yarı Gol Alt/Üst :: Dep İY {s} Alt", (iag < v).sum(), odd(test_row, f"Deplasman İlk Yarı Gol Alt/Üst :: Dep İY {s} Alt"))
        push_if_value(rows, test_row, f"Deplasman İlk Yarı Gol Alt/Üst :: Dep İY {s} Üst", (iag >= v).sum(), odd(test_row, f"Deplasman İlk Yarı Gol Alt/Üst :: Dep İY {s} Üst"))

    # Ev Sahibi alt/üst (full & first half)
    ev_thr_full = [0.5, 1.5, 2.5, 3.5, 4.5]
    ev_thr_lbl  = ["0,5","1,5","2,5","3,5","4,5"]
    for v, s in zip(ev_thr_full, ev_thr_lbl):
        push_if_value(rows, test_row, f"Ev Sahibi Gol Alt/Üst :: Ev {s} Alt", (hg < v).sum(), odd(test_row, f"Ev Sahibi Gol Alt/Üst :: Ev {s} Alt"))
        push_if_value(rows, test_row, f"Ev Sahibi Gol Alt/Üst :: Ev {s} Üst", (hg >= v).sum(), odd(test_row, f"Ev Sahibi Gol Alt/Üst :: Ev {s} Üst"))

    push_if_value(rows, test_row, "Ev Sahibi Gol Yemeden Kazanır mı? :: Evet", ((hg > ag) & (ag == 0)).sum(), odd(test_row, "Ev Sahibi Gol Yemeden Kazanır mı? :: Evet"))
    push_if_value(rows, test_row, "Ev Sahibi Gol Yemeden Kazanır mı? :: Hayır", (~((hg > ag) & (ag == 0))).sum(), odd(test_row, "Ev Sahibi Gol Yemeden Kazanır mı? :: Hayır"))

    push_if_value(rows, test_row, "Ev Sahibi Hangi Yarıda Daha Fazla Gol Atar? :: 1. Yarı", (ihg > shg).sum(), odd(test_row, "Ev Sahibi Hangi Yarıda Daha Fazla Gol Atar? :: 1. Yarı"))
    push_if_value(rows, test_row, "Ev Sahibi Hangi Yarıda Daha Fazla Gol Atar? :: 2. Yarı", (shg > ihg).sum(), odd(test_row, "Ev Sahibi Hangi Yarıda Daha Fazla Gol Atar? :: 2. Yarı"))
    push_if_value(rows, test_row, "Ev Sahibi Hangi Yarıda Daha Fazla Gol Atar? :: Eşit", (shg == ihg).sum(), odd(test_row, "Ev Sahibi Hangi Yarıda Daha Fazla Gol Atar? :: Eşit"))

    ev_iy_thr = [0.5, 1.5, 2.5]
    ev_iy_lbl = ["0,5","1,5","2,5"]
    for v, s in zip(ev_iy_thr, ev_iy_lbl):
        push_if_value(rows, test_row, f"Ev Sahibi İlk Yarı Gol Alt/Üst :: Ev İY {s} Alt", (ihg < v).sum(), odd(test_row, f"Ev Sahibi İlk Yarı Gol Alt/Üst :: Ev İY {s} Alt"))
        push_if_value(rows, test_row, f"Ev Sahibi İlk Yarı Gol Alt/Üst :: Ev İY {s} Üst", (ihg >= v).sum(), odd(test_row, f"Ev Sahibi İlk Yarı Gol Alt/Üst :: Ev İY {s} Üst"))

    # Fark bahisleri
    gd = hg - ag
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Berabere", (gd == 0).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Berabere"))
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Dep 1 Fark", (gd == -1).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Dep 1 Fark"))
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Dep 2 Fark", (gd == -2).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Dep 2 Fark"))
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Dep 3+ Fark", (gd <= -3).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Dep 3+ Fark"))
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Ev 1 Fark",  (gd == 1).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Ev 1 Fark"))
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Ev 2 Fark",  (gd == 2).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Ev 2 Fark"))
    push_if_value(rows, test_row, "Hangi Takım Kaç Farkla Kazanır :: Ev 3+ Fark", (gd >= 3).sum(), odd(test_row, "Hangi Takım Kaç Farkla Kazanır :: Ev 3+ Fark"))

    # Hangi yarı daha fazla gol?
    push_if_value(rows, test_row, "Hangi Yarıda Daha Fazla Gol Atılır? :: 1. Yarı", (itg > (tg - itg)).sum(), odd(test_row, "Hangi Yarıda Daha Fazla Gol Atılır? :: 1. Yarı"))
    push_if_value(rows, test_row, "Hangi Yarıda Daha Fazla Gol Atılır? :: 2. Yarı", ((tg - itg) > itg).sum(), odd(test_row, "Hangi Yarıda Daha Fazla Gol Atılır? :: 2. Yarı"))
    push_if_value(rows, test_row, "Hangi Yarıda Daha Fazla Gol Atılır? :: Eşit", (itg == (tg - itg)).sum(), odd(test_row, "Hangi Yarıda Daha Fazla Gol Atılır? :: Eşit"))

    # Her iki yarıda 1.5 alt/üst
    push_if_value(rows, test_row, "Her İki Yarıda da 1.5 Gol Alt Olur mu? :: Evet", ((itg < 1.5) & ((tg - itg) < 1.5)).sum(), odd(test_row, "Her İki Yarıda da 1.5 Gol Alt Olur mu? :: Evet"))
    push_if_value(rows, test_row, "Her İki Yarıda da 1.5 Gol Alt Olur mu? :: Hayır", (~((itg < 1.5) & ((tg - itg) < 1.5))).sum(), odd(test_row, "Her İki Yarıda da 1.5 Gol Alt Olur mu? :: Hayır"))
    push_if_value(rows, test_row, "Her İki Yarıda da 1.5 Gol Üst Olur mu? :: Evet", ((itg >= 1.5) & ((tg - itg) >= 1.5)).sum(), odd(test_row, "Her İki Yarıda da 1.5 Gol Üst Olur mu? :: Evet"))
    push_if_value(rows, test_row, "Her İki Yarıda da 1.5 Gol Üst Olur mu? :: Hayır", (~((itg >= 1.5) & ((tg - itg) >= 1.5))).sum(), odd(test_row, "Her İki Yarıda da 1.5 Gol Üst Olur mu? :: Hayır"))

    # KG
    kg_var = (hg > 0) & (ag > 0)
    kg_yok = ~kg_var
    push_if_value(rows, test_row, "Karşılıklı Gol :: KG Var", kg_var.sum(), odd(test_row, "Karşılıklı Gol :: KG Var"))
    push_if_value(rows, test_row, "Karşılıklı Gol :: KG Yok", kg_yok.sum(), odd(test_row, "Karşılıklı Gol :: KG Yok"))

    # KG + 2,5 kombinasyon
    push_if_value(rows, test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Alt ve KG Var", ((tg < 2.5) & kg_var).sum(), odd(test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Alt ve KG Var"))
    push_if_value(rows, test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Alt ve KG Yok", ((tg < 2.5) & kg_yok).sum(), odd(test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Alt ve KG Yok"))
    push_if_value(rows, test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Üst ve KG Var", ((tg >= 2.5) & kg_var).sum(), odd(test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Üst ve KG Var"))
    push_if_value(rows, test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Üst ve KG Yok", ((tg >= 2.5) & kg_yok).sum(), odd(test_row, "Karşılıklı Gol ve 2,5 Gol Alt/Üst :: 2,5 Üst ve KG Yok"))

    # Toplam gol alt/üst
    total_thr = [0.5,1.5,2.5,3.5,4.5,5.5,6.5]
    total_lbl = ["0,5","1,5","2,5","3,5","4,5","5,5","6,5"]
    for v, s in zip(total_thr, total_lbl):
        push_if_value(rows, test_row, f"Toplam Gol Alt/Üst :: {s} Alt", (tg < v).sum(), odd(test_row, f"Toplam Gol Alt/Üst :: {s} Alt"))
        push_if_value(rows, test_row, f"Toplam Gol Alt/Üst :: {s} Üst", (tg >= v).sum(), odd(test_row, f"Toplam Gol Alt/Üst :: {s} Üst"))

    # Toplam gol aralığı
    push_if_value(rows, test_row, "Toplam Gol Aralığı :: 0-1 Gol", (tg <= 1).sum(), odd(test_row, "Toplam Gol Aralığı :: 0-1 Gol"))
    push_if_value(rows, test_row, "Toplam Gol Aralığı :: 2-3 Gol", ((tg >= 2) & (tg <= 3)).sum(), odd(test_row, "Toplam Gol Aralığı :: 2-3 Gol"))
    push_if_value(rows, test_row, "Toplam Gol Aralığı :: 4-5 Gol", ((tg >= 4) & (tg <= 5)).sum(), odd(test_row, "Toplam Gol Aralığı :: 4-5 Gol"))
    push_if_value(rows, test_row, "Toplam Gol Aralığı :: 6+ Gol", (tg >= 6).sum(), odd(test_row, "Toplam Gol Aralığı :: 6+ Gol"))

    # Tek/Çift (full & first half)
    push_if_value(rows, test_row, "Toplam Gol Tek/Çift :: Tek",  (tg % 2 == 1).sum(), odd(test_row, "Toplam Gol Tek/Çift :: Tek"))
    push_if_value(rows, test_row, "Toplam Gol Tek/Çift :: Çift", (tg % 2 == 0).sum(), odd(test_row, "Toplam Gol Tek/Çift :: Çift"))

    # Çifte Şans
    push_if_value(rows, test_row, "Çifte Şans :: ÇŞ 1-2", (hg != ag).sum(), odd(test_row, "Çifte Şans :: ÇŞ 1-2"))
    push_if_value(rows, test_row, "Çifte Şans :: ÇŞ 1-X", (hg >= ag).sum(), odd(test_row, "Çifte Şans :: ÇŞ 1-X"))
    push_if_value(rows, test_row, "Çifte Şans :: ÇŞ X-2", (ag >= hg).sum(), odd(test_row, "Çifte Şans :: ÇŞ X-2"))

    # 2. yarı KG & sonuç
    push_if_value(rows, test_row, "İkinci Yarı Karşılıklı Gol :: 2.Y KG Var", ((shg > 0) & (sag > 0)).sum(), odd(test_row, "İkinci Yarı Karşılıklı Gol :: 2.Y KG Var"))
    push_if_value(rows, test_row, "İkinci Yarı Karşılıklı Gol :: 2.Y KG Yok", (~((shg > 0) & (sag > 0))).sum(), odd(test_row, "İkinci Yarı Karşılıklı Gol :: 2.Y KG Yok"))

    push_if_value(rows, test_row, "İkinci Yarı Sonucu :: 2.Y 1", (shg > sag).sum(), odd(test_row, "İkinci Yarı Sonucu :: 2.Y 1"))
    push_if_value(rows, test_row, "İkinci Yarı Sonucu :: 2.Y 2", (sag > shg).sum(), odd(test_row, "İkinci Yarı Sonucu :: 2.Y 2"))
    push_if_value(rows, test_row, "İkinci Yarı Sonucu :: 2.Y X", (sag == shg).sum(), odd(test_row, "İkinci Yarı Sonucu :: 2.Y X"))

    # İlk yarı / maç sonucu (1/1, 1/2, 1/X, ...)
    iy1 = ihg > iag
    iy2 = iag > ihg
    iyx = ihg == iag
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: 1/1", (iy1 & (hg > ag)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: 1/1"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: 1/2", (iy1 & (ag > hg)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: 1/2"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: 1/X", (iy1 & (hg == ag)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: 1/X"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: 2/1", (iy2 & (hg > ag)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: 2/1"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: 2/2", (iy2 & (ag > hg)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: 2/2"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: 2/X", (iy2 & (hg == ag)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: 2/X"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: X/1", (iyx & (hg > ag)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: X/1"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: X/2", (iyx & (ag > hg)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: X/2"))
    push_if_value(rows, test_row, "İlk Yarı / Maç Sonucu :: X/X", (iyx & (hg == ag)).sum(), odd(test_row, "İlk Yarı / Maç Sonucu :: X/X"))

    # İlk yarı toplam gol alt/üst
    iy_thr = [0.5,1.5,2.5,4.5]
    iy_lbl = ["0,5","1,5","2,5","4,5"]
    for v, s in zip(iy_thr, iy_lbl):
        push_if_value(rows, test_row, f"İlk Yarı Gol Alt/Üst :: İY {s} Alt", (itg < v).sum(), odd(test_row, f"İlk Yarı Gol Alt/Üst :: İY {s} Alt"))
        push_if_value(rows, test_row, f"İlk Yarı Gol Alt/Üst :: İY {s} Üst", (itg >= v).sum(), odd(test_row, f"İlk Yarı Gol Alt/Üst :: İY {s} Üst"))

    # İlk yarı tek/çift
    push_if_value(rows, test_row, "İlk Yarı Gol Tek/Çift :: İY Tek",  (itg % 2 == 1).sum(), odd(test_row, "İlk Yarı Gol Tek/Çift :: İY Tek"))
    push_if_value(rows, test_row, "İlk Yarı Gol Tek/Çift :: İY Çift", (itg % 2 == 0).sum(), odd(test_row, "İlk Yarı Gol Tek/Çift :: İY Çift"))

    # İlk yarı KG
    iy_kg_var = (ihg > 0) & (iag > 0)
    iy_kg_yok = ~iy_kg_var
    push_if_value(rows, test_row, "İlk Yarı Karşılıklı Gol :: İY KG Var", iy_kg_var.sum(), odd(test_row, "İlk Yarı Karşılıklı Gol :: İY KG Var"))
    push_if_value(rows, test_row, "İlk Yarı Karşılıklı Gol :: İY KG Yok", iy_kg_yok.sum(), odd(test_row, "İlk Yarı Karşılıklı Gol :: İY KG Yok"))

    # İlk yarı skoru (enumeration)
    iy_scores = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    matched = np.zeros(len(neigh), dtype=bool)
    for h,a in iy_scores:
        lab = f"İlk Yarı Skoru :: {h}-{a}"
        mask = (ihg == h) & (iag == a)
        matched |= mask
        push_if_value(rows, test_row, lab, mask.sum(), odd(test_row, lab))
    push_if_value(rows, test_row, "İlk Yarı Skoru :: diğer", (~matched).sum(), odd(test_row, "İlk Yarı Skoru :: diğer"))

    # Maç Skoru enumeration (your original exhaustive set)
    possible_scores = [
        (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
        (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
        (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
        (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
        (4,0),(4,1),(4,2),(4,3),(4,4),
        (5,0),(5,1),(5,2),(5,3),(5,4),
        (6,0),(6,1),(6,2)
    ]
    matched = np.zeros(len(neigh), dtype=bool)
    for h,a in possible_scores:
        for sep in ("-",
                    ":"):  # keep both variants as in your code
            lab = f"Maç Skoru :: {h}{sep}{a}"
            mask = (hg == h) & (ag == a)
            matched |= mask
            push_if_value(rows, test_row, lab, mask.sum(), odd(test_row, lab))
    push_if_value(rows, test_row, "Maç Skoru :: diğer", (~matched).sum(), odd(test_row, "Maç Skoru :: diğer"))

    # Maç Sonucu + Alt/Üst combined
    combo_thr = [1.5,2.5,3.5,4.5]
    combo_lbl = ["1,5","2,5","3,5","4,5"]
    for v, s in zip(combo_thr, combo_lbl):
        push_if_value(rows, test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 1 ve {s} Alt", ((ms1) & (tg < v)).sum(), odd(test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 1 ve {s} Alt"))
        push_if_value(rows, test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 1 ve {s} Üst", ((ms1) & (tg >= v)).sum(), odd(test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 1 ve {s} Üst"))
        push_if_value(rows, test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 2 ve {s} Alt", ((ms2) & (tg < v)).sum(), odd(test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 2 ve {s} Alt"))
        push_if_value(rows, test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 2 ve {s} Üst", ((ms2) & (tg >= v)).sum(), odd(test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS 2 ve {s} Üst"))
        push_if_value(rows, test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS X ve {s} Alt", ((msx) & (tg < v)).sum(), odd(test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS X ve {s} Alt"))
        push_if_value(rows, test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS X ve {s} Üst", ((msx) & (tg >= v)).sum(), odd(test_row, f"Maç Sonucu ve {s} Gol Alt/Üst :: MS X ve {s} Üst"))

    # Maç Sonucu + KG
    push_if_value(rows, test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 1 ve Var", (ms1 & kg_var).sum(), odd(test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 1 ve Var"))
    push_if_value(rows, test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 1 ve Yok", (ms1 & (~kg_var)).sum(), odd(test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 1 ve Yok"))
    push_if_value(rows, test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 2 ve Var", (ms2 & kg_var).sum(), odd(test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 2 ve Var"))
    push_if_value(rows, test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 2 ve Yok", (ms2 & (~kg_var)).sum(), odd(test_row, "Maç Sonucu ve Karşılıklı Gol :: MS 2 ve Yok"))
    push_if_value(rows, test_row, "Maç Sonucu ve Karşılıklı Gol :: MS X ve Var", (msx & kg_var).sum(), odd(test_row, "Maç Sonucu ve Karşılıklı Gol :: MS X ve Var"))
    push_if_value(rows, test_row, "Maç Sonucu ve Karşılıklı Gol :: MS X ve Yok", (msx & (~kg_var)).sum(), odd(test_row, "Maç Sonucu ve Karşılıklı Gol :: MS X ve Yok"))

# -------- main loop (vectorized distances, no iterrows) --------
value_rows = []

for i in range(len(test_df)):
    test_row = test_df.iloc[i]

    # compute distances test i -> all train
    dists = nan_euclidean_to_train(X[i], X_nan[i])

    # argpartition to get indices of TOP_K smallest distances
    if np.isfinite(dists).sum() == 0:
        # skip if nothing comparable
        continue

    k = min(TOP_K, np.isfinite(dists).sum())
    top_idx = np.argpartition(dists, k - 1)[:k]

    # build neighbor DF once
    neigh = train_df.iloc[top_idx][[
        "totalHomeGoal", "totalAwayGoal",
        "firstHalfHomeGoal", "firstHalfAwayGoal"
    ]].reset_index(drop=True)

    # compute all bet counts + push value rows
    compute_bet_counts_and_push(value_rows, test_row, neigh)

# -------- build DataFrame once --------
value_bets = pd.DataFrame(value_rows, columns=[
    "match_date","match_time","match_id","hometeam","awayteam",
    "bet_name","probability","odds"
])

# Persist raw list
value_bets.to_csv(f"value_bets_{OUT_PREFIX}.csv", index=False)

# -------- Post-processing (EV, Kelly, bankroll) --------
svb = value_bets.copy()
svb["EV"] = (svb["probability"] * svb["odds"]) / 100.0

p = svb["probability"] / 100.0
b = svb["odds"] - 1.0
q = 1.0 - p
svb["Kelly"] = ((b * p - q) / b).clip(lower=0.0).fillna(0.0)
svb["stake"] = svb["Kelly"] * 100.0
svb["bankroll_if_win"]  = 1.0 + svb["Kelly"] * (svb["odds"] - 1.0)
svb["bankroll_if_lose"] = 1.0 - svb["Kelly"]
svb["expected_bankroll"] = p * svb["bankroll_if_win"] + q * svb["bankroll_if_lose"]

# Top by probability (per match)
value_bets_top_prob = (
    svb.sort_values("probability", ascending=False)
       .groupby("match_id", as_index=False)
       .first()
)
# Top by expected bankroll (per match)
value_bets_top_bankroll = (
    svb.sort_values("expected_bankroll", ascending=False)
       .groupby("match_id", as_index=False)
       .first()
)

# Time-ordered outputs
sorted_vals_prob_time = value_bets_top_prob.sort_values(["match_date","match_time"], ascending=True)
sorted_vals_prob_time.to_csv(f"value_bets_by_prob_{OUT_PREFIX}.csv", index=False)

sorted_vals_bankroll_time = value_bets_top_bankroll.sort_values(["match_date","match_time"], ascending=True)
sorted_vals_bankroll_time.to_csv(f"value_bets_by_bankroll_{OUT_PREFIX}.csv", index=False)

if PRINT_SAMPLE:
    print(svb.sort_values("EV", ascending=False).head(5))
