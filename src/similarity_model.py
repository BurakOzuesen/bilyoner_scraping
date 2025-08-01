from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def find_similar_matches(df, target_row, feature_cols, k=100):
    """
    Belirli feature'lara göre hedef maça en çok benzeyen K maçı döndürür
    """
    feature_matrix = df[feature_cols].fillna(0).astype(float)
    target_vector = target_row[feature_cols].fillna(0).astype(float).values.reshape(1, -1)

    similarities = cosine_similarity(feature_matrix, target_vector).flatten()
    df["similarity_score"] = similarities

    top_matches = df.sort_values(by="similarity_score", ascending=False).head(k)
    return top_matches.drop(columns=["similarity_score"])


def estimate_probabilities(similar_matches):
    """
    Benzer maçların skorlarına göre MS1, MS2, MS X olasılıklarını tahmin eder
    """
    outcome_counts = {"MS1": 0, "MS2": 0, "MS X": 0}
    for _, row in similar_matches.iterrows():
        home = row["totalHomeGoal"]
        away = row["totalAwayGoal"]
        if pd.isna(home) or pd.isna(away):
            continue
        if home > away:
            outcome_counts["MS1"] += 1
        elif home < away:
            outcome_counts["MS2"] += 1
        else:
            outcome_counts["MS X"] += 1

    total = sum(outcome_counts.values())
    if total == 0:
        return None  # tüm skorlar eksikse
    probs = {k: round(v / total, 2) for k, v in outcome_counts.items()}
    return probs

def estimate_probabilities(similar_matches):
    """
    Benzer maçların skorlarına göre MS1, MS2, MS X olasılıklarını tahmin eder
    """
    outcome_counts = {"MS1": 0, "MS2": 0, "MS X": 0}
    for _, row in similar_matches.iterrows():
        home = row["totalHomeGoal"]
        away = row["totalAwayGoal"]
        if pd.isna(home) or pd.isna(away):
            continue
        if home > away:
            outcome_counts["MS1"] += 1
        elif home < away:
            outcome_counts["MS2"] += 1
        else:
            outcome_counts["MS X"] += 1

    total = sum(outcome_counts.values())
    if total == 0:
        return None  # tüm skorlar eksikse
    probs = {k: round(v / total, 2) for k, v in outcome_counts.items()}
    return probs


def value_bet_decision(prob, odds):
    """
    Olasılığa göre EV hesapla, value bet olup olmadığını döndür
    """
    if prob is None or odds is None or odds == 0:
        return None, None
    ev = round((prob * odds), 2)
    is_value = ev > 1
    return ev, is_value


def run_similarity_prediction(df, match_id, feature_cols, k=100):
    """
    Belirtilen match_id için benzer maçlara dayalı olasılık tahmini ve value bet kararı döndürür.
    """
    if match_id not in df["match_id"].values:
        print(f"⛔ match_id {match_id} veri setinde bulunamadı.")
        return None

    target_row = df[df["match_id"] == match_id].iloc[0]
    similar = find_similar_matches(df, target_row, feature_cols, k=k)
    probs = estimate_probabilities(similar)

    if probs is None:
        return None

    results = {}
    for outcome, col in zip(["MS1", "MS2", "MS X"], feature_cols):
        ev, is_val = value_bet_decision(probs[outcome], target_row[col])
        results[outcome] = {
            "prob": probs[outcome],
            "odds": target_row[col],
            "ev": ev,
            "is_value": is_val
        }
    return results
