# import pandas as pd

# CSV_PATH = "match_odds_wide.csv"
# KEY_COLS = [
#     "match_date", "match_time", "tournament", "match_id",
#     "homeTeam", "awayTeam",
#     "firstHalfHomeGoal", "firstHalfAwayGoal",
#     "totalHomeGoal", "totalAwayGoal",
#     "homeCorner", "awayCorner"
# ]

# # CSV'yi oku
# df = pd.read_csv(CSV_PATH)

# # KEY_COLS harici tüm sütunları boşalt
# cols_to_clear = [col for col in df.columns if col not in KEY_COLS]
# df[cols_to_clear] = pd.NA

# # Güncellenmiş DataFrame'i kaydet
# df.to_csv(CSV_PATH, index=False)

# print(f"{len(cols_to_clear)} sütun boşaltıldı ve dosya kaydedildi.")


import pandas as pd

CSV_PATH = "match_odds_wide.csv"
KEY_COLS = [
    "match_date", "match_time", "tournament", "match_id",
    "homeTeam", "awayTeam",
    "firstHalfHomeGoal", "firstHalfAwayGoal",
    "totalHomeGoal", "totalAwayGoal",
    "homeCorner", "awayCorner"
]

# CSV'yi oku
df = pd.read_csv(CSV_PATH)

# Sadece KEY_COLS sütunlarını tut
df = df[KEY_COLS]

# Temizlenmiş DataFrame'i dosyaya yaz
df.to_csv(CSV_PATH, index=False)

print("KEY_COLS dışındaki tüm sütunlar silindi ve dosya kaydedildi.")
