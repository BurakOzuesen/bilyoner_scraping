import pandas as pd

def calculate_ev(probability, odds):
    """
    Beklenen Değer (EV) = (olasılık * oran) - (1 - olasılık)
    EV > 0 ise value bet olabilir
    """
    if probability is None or odds is None:
        return None
    return round((probability * odds) - (1 - probability), 4)

def implied_probability(odds):
    """
    Orandan implied probability (ters oran) hesaplama
    """
    if odds is None or odds == 0:
        return None
    return round(1 / odds, 4)

def run_value_analysis(df, target_columns):
    """
    Verilen oran sütunları için implied probability, EV ve value bet etiketi hesaplar.
    target_columns: Örn. ["Maç Sonucu :: MS 1", "Maç Sonucu :: MS 2", "Maç Sonucu :: MS X"]
    """
    for col in target_columns:
        ev_col = f"EV :: {col}"
        prob_col = f"Implied Prob :: {col}"
        is_value_col = f"is_value :: {col}"

        df[prob_col] = df[col].apply(implied_probability)
        df[ev_col] = df.apply(lambda row: calculate_ev(row[prob_col], row[col]), axis=1)
        df[is_value_col] = df[ev_col] > 0

        print(f"📊 Value analizi işlendi: {col}")

    return df
