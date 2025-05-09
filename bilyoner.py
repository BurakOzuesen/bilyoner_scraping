import requests
import pandas as pd
import random
from collections import defaultdict

# 1. Maç listesini çek
url_list = "https://www.bilyoner.com/api/v3/mobile/aggregator/gamelist/all/v1"
params = {"tabType": 1, "bulletinType": 2, "liveEventsEnabledForPreBulletin": "true"}
headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(url_list, params=params, headers=headers)
resp.raise_for_status()
data = resp.json()

# 2. event_items sözlüğünü al (eventId -> maç bilgisi)
# JSON yapısına göre data["data"]["eventList"] listesi var
event_list = data["data"]["eventList"]
event_items = { e["eventId"]: e for e in event_list }

# 3. Rastgele 100 maç için smart-analysis çek ve sözlüğe kaydet
smart_analysis_results = {}
for _ in range(100):
    match = random.choice(list(event_items.values()))
    maç_id = match["eventId"]
    url_sa = f"https://www.bilyoner.com/api/mobile/match-card/v2/{maç_id}/smart-analysis"
    r = requests.get(url_sa, headers=headers)
    smart_analysis_results[maç_id] = r.json() if r.ok else None

# 4. Yorum içeren market’lerden tüm outcome’ları topla
all_odds = []
for eid, analysis in smart_analysis_results.items():
    if not analysis: 
        continue
    for m in analysis.get("markets", []):
        if "comment" in m:
            for outcome in m["markets"]:
                all_odds.append({
                    "event_id": eid,
                    "marketName": m["marketName"],
                    "label": outcome["label"],
                    "fixedOdds": outcome["fixedOddsWeb"],
                    "currentOdds": outcome["currentOddsWeb"]
                })

# 5. event_id + marketName’e göre grupla ve lowest currentOdds’u seç
groups = defaultdict(list)
for odd in all_odds:
    groups[(odd["event_id"], odd["marketName"])].append(odd)
selected = [min(v, key=lambda x: x["currentOdds"]) for v in groups.values()]

# 6. Satırları hazırla: maç bilgisi + seçilen odd
rows = []
for odd in selected:
    eid = odd["event_id"]
    match = event_items.get(eid)
    if not match:
        continue
    rows.append({
        "gün":       match.get("strd"),
        "saat":      match.get("strt"),
        "turnuva":   match.get("slgn"),
        "ev":        match.get("htn"),
        "dep":       match.get("atn"),
        "maç_id":    eid,
        "marketName":odd["marketName"],
        "label":     odd["label"],
        "fixedOdds": odd["fixedOdds"],
        "currentOdds": odd["currentOdds"]
    })

# 7. DataFrame oluşturup CSV & Excel kaydet
df = pd.DataFrame(rows, columns=[
    "gün","saat","turnuva","ev","dep","maç_id",
    "marketName","label","fixedOdds","currentOdds"
])
df.to_csv("combined_matches.csv", index=False, encoding="utf-8-sig")
df.to_excel("combined_matches.xlsx", index=False)

print("Oluşan tablo:")
print(df)
print("\nDosyalar kaydedildi: combined_matches.csv, combined_matches.xlsx")
