import pandas as pd
import os
from datetime import datetime

def filter_missing_ms(df, snapshot_dir="data/processed/"):
    """
    'Maç Sonucu :: MS 1', 'MS 2', 'MS X' oranlarının tamamı NaN olan maçları siler.
    Geri kalanları snapshot olarak kaydeder.
    """
    required_cols = ["Maç Sonucu :: MS 1", "Maç Sonucu :: MS 2", "Maç Sonucu :: MS X"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' kolonu bulunamadı.")

    before = len(df)
    filtered_df = df.dropna(subset=required_cols, how="all")
    after = len(filtered_df)
    removed = before - after

    today = datetime.today().strftime("%Y%m%d")
    os.makedirs(snapshot_dir, exist_ok=True)
    output_path = os.path.join(snapshot_dir, f"match_odds_filtered_ms_{today}.csv")
    filtered_df.to_csv(output_path, index=False)

    print(f"✅ MS oranı olmayan {removed} maç silindi. Kalan: {after}")
    print(f"📦 Kaydedildi: {output_path}")

    return filtered_df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/match_odds_wide_20250801.csv")
    filter_missing_ms(df)