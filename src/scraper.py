import pandas as pd
import os
from datetime import datetime
import time
import requests
from tqdm import tqdm  # ilerleme göstergesi için

def get_score_by_code(match_id):
    """
    Bilyoner API'den verilen maç kodu ile skor bilgisini çeker.
    """
    try:
        url = f"https://www.bilyoner.com/api/mobile/match-card/v2/{match_id}/status?sgSportTypeId=1"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Varsayılan None değerleri
        fh_home = fh_away = ft_home = ft_away = None

        for score_entry in data.get("score", []):
            if score_entry.get("scoreName") == "SOCCER_FIRST_HALF":
                fh_home = int(score_entry.get("homeScore", "0"))
                fh_away = int(score_entry.get("awayScore", "0"))
            elif score_entry.get("scoreName") == "SOCCER_END_SCORE":
                ft_home = int(score_entry.get("homeScore", "0"))
                ft_away = int(score_entry.get("awayScore", "0"))

        return {
            "firstHalfHomeGoal": fh_home,
            "firstHalfAwayGoal": fh_away,
            "totalHomeGoal": ft_home,
            "totalAwayGoal": ft_away
        }

    except Exception as e:
        log_msg = f"[HATA] Kod {match_id} için skor alınamadı: {e}"
        print(log_msg)
        with open("logs/score_errors.txt", "a") as f:
            f.write(log_msg + "\n")
        return {
            "firstHalfHomeGoal": None,
            "firstHalfAwayGoal": None,
            "totalHomeGoal": None,
            "totalAwayGoal": None
        }

def refresh_scores(input_path="data/processed/match_odds_wide.csv",
                   output_dir="data/processed/",
                   id_column="match_id",
                   sleep_time=0.3):
    """
    Skorları siler ve her maç için yeniden getirir.
    Güncellenmiş dosya tarihli olarak kayıt edilir.
    """
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Giriş dosyası bulunamadı: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"📥 Veri yüklendi: {len(df)} maç")

    # Skorları temizle
    score_columns = [
        "firstHalfHomeGoal", "firstHalfAwayGoal",
        "totalHomeGoal", "totalAwayGoal"
    ]
    df_clean = df.drop(columns=score_columns, errors="ignore")

    # Skorları çek
    scores = []
    print("🔄 Skorlar çekiliyor...")
    for i, code in enumerate(tqdm(df_clean[id_column], desc="Maç işleniyor", unit="maç")):
        score = get_score_by_code(code)
        scores.append(score)
        # time.sleep(sleep_time)

    # Skorları ekle
    scores_df = pd.DataFrame(scores)
    df_final = pd.concat([df_clean.reset_index(drop=True), scores_df], axis=1)

    # Kaydet
    today = datetime.today().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"match_odds_wide_{today}.csv")
    df_final.to_csv(output_file, index=False)

    print(f"\n✅ Skorlar güncellendi: {output_file}")
    print(f"📊 Güncellenen maç sayısı: {len(df_final)}")
    print(f"🕒 Snapshot zamanı: {today}")
    print(f"📝 Hatalar için: logs/score_errors.txt")

if __name__ == "__main__":
    refresh_scores()