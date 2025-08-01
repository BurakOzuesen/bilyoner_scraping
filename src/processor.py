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
    print(f"📆 Kaydedildi: {output_path}")

    return filtered_df

def detect_useless_columns(df, threshold_null=0.99, threshold_unique=1):
    """
    Yüksek oranda null olan veya tüm satırlarda aynı değeri içeren sütunları tespit eder.
    threshold_null: % olarak null oranı sınırı (0.99 = %99 ve üzeri null ise "gereksiz")
    threshold_unique: benzersiz değer sayısı (1 = sabit sütun)
    """
    total_rows = len(df)
    report = []

    for col in df.columns:
        null_ratio = df[col].isna().mean()
        unique_count = df[col].nunique(dropna=True)

        if null_ratio >= threshold_null or unique_count <= threshold_unique:
            report.append({
                "column": col,
                "null_ratio": round(null_ratio, 4),
                "unique_count": unique_count
            })

    useless_df = pd.DataFrame(report)
    useless_df.sort_values(by="null_ratio", ascending=False, inplace=True)

    print(f"🔍 Tespit edilen gereksiz/sabit sütun sayısı: {len(useless_df)}")
    return useless_df

def nullify_empty_markets(df, useless_df, threshold=0.95):
    """
    Sütun büyük oranda NaN ama tamamen boş değilse, veri olmayan hücreleri NaN yapar.
    Sütun kalır, ama hücreler temizlenmiş olur.
    """
    columns_to_nullify = useless_df[
        (useless_df["null_ratio"] >= threshold) &
        (useless_df["null_ratio"] < 1.0)
    ]["column"].tolist()

    for col in columns_to_nullify:
        df[col] = df[col].where(df[col].notna(), None)

    print(f"🧽 Hücresel temizleme yapıldı: {len(columns_to_nullify)} sütun")
    return df

def drop_useless_columns(df, useless_df, threshold=0.99):
    """
    Sütun %99+ NaN ise veya sadece tek bir unique değer taşıyorsa sütunu tamamen siler.
    """
    columns_to_drop = useless_df[
        (useless_df["null_ratio"] >= threshold) |
        (useless_df["unique_count"] <= 1)
    ]["column"].tolist()

    df_cleaned = df.drop(columns=columns_to_drop, errors="ignore")
    print(f"🚹 Tamamen silinen sütun sayısı: {len(columns_to_drop)}")
    return df_cleaned

if __name__ == "__main__":
    df = pd.read_csv("data/processed/match_odds_wide_20250801.csv")
    df = filter_missing_ms(df)
    useless = detect_useless_columns(df)
    df = nullify_empty_markets(df, useless)
    df = drop_useless_columns(df, useless)

    today = datetime.today().strftime("%Y%m%d")
    output_path = f"data/processed/match_odds_cleaned_{today}.csv"
    df.to_csv(output_path, index=False)
    print(f"📅 Temizlenmiş veri kaydedildi: {output_path}")
