# 🎯 Bilyoner Value Bet Projesi - Task Listesi

Bu dosya, oran tahmini projesinde yapılacak tüm işleri kategorize ederek takip etmek için hazırlanmıştır.

---

## ✅ A. VERİ TEMİZLİĞİ & HAZIRLIK

- [ ] **A1: Tüm skorları sil ve yeniden kaydet**  
  _Açıklama_: Eski skorlar güvenilmez olabilir. Tüm skor verileri silinip güncel maç ID’leriyle yeniden çekilecek ve dosyaya/DB'ye kaydedilecek.  
  ✨ Hedef fonksiyon: `refresh_scores()`

- [ ] **A2: MS oranı olmayan maçları sil**  
  _Açıklama_: MS oranı olmayan maçlar eksik veri içeriyor olabilir. Bunlar canlıdan gelen incomplete veriler olabilir.  
  ✨ Hedef fonksiyon: `filter_missing_ms()`

- [ ] **A3: Fazlalık sütunları tespit et**  
  _Açıklama_: Tüm oran dataframe'inde sadece veri içeren (non-null, varyansı yüksek) sütunlar belirlenecek.  
  ✨ Hedef fonksiyon: `detect_useless_columns()`

- [ ] **A4: Oranı olmayan ama sütunu dolu maçların hücrelerini sil**  
  _Açıklama_: Örneğin “Çifte Şans” sütunu var ama bu maçta oynanmamış; bu gibi değerleri NaN yap.  
  ✨ Hedef fonksiyon: `nullify_empty_markets()`

- [ ] **A5: Fazlalık sütunları tamamen sil**  
  _Açıklama_: Önceki adımdaki analiz sonuçlarına göre hiçbir zaman veri içermeyen sütunlar silinecek.  
  ✨ Hedef fonksiyon: `drop_useless_columns()`

---

## 🚀 B. PERFORMANS OPTİMİZASYONU

- [ ] **B1: Insert işlemlerini hızlandır**  
  _Açıklama_: `df.to_csv` / `to_sql` işlemleri yavaşsa bulk insert, Parquet veya `feather` gibi yöntemler test edilecek.  
  ✨ Hedef fonksiyon: `save_data(df, method="auto")`

- [ ] **B2: Lambda ve apply kullanımlarını vektörel hale getir**  
  _Açıklama_: Tüm lambda’lar NumPy vektörleri veya `np.where` gibi yöntemlere geçirilecek.  
  ✨ Hedef fonksiyon: `vectorize_processing()`

- [ ] **B3: Zamanlama loglaması eklensin**  
  _Açıklama_: Her fonksiyonun çalışma süresi loglansın. Performans darboğazı varsa tespit edilsin.  
  ✨ Hedef fonksiyon: `@timing_logger`

---

## 🧱 C. YAPISAL DÜZENLEME & MODÜLERLİK

- [ ] **C1: Kodları `.py` dosyalarına böl (scraper.py, predictor.py)**  
  _Açıklama_: Notebook'lar modül haline getirilecek. Tek tuşla yeniden çalıştırılabilir olacak.  
  ✨ Hedef klasörler: `src/scraper.py`, `src/predictor.py`

- [ ] **C2: Ortak işlemler fonksiyonlaştırılacak**  
  _Açıklama_: `save_to_csv`, `clean_df`, `filter_df` gibi tekrar eden işlemler modül fonksiyonuna dönüşecek.  
  ✨ Hedef dosya: `utils.py`

- [ ] **C3: Value bet analizlerini fonksiyonlaştır**  
  _Açıklama_: Bahis türüne göre oranlardan implied value hesaplayan fonksiyon yazılacak.  
  ✨ Örn: `calculate_value(oran, prob)` veya `calculate_ev(...)`

---

## 🧪 D. TEST & DOĞRULAMA (sonraki aşama)

- [ ] **D1: Test dataset'i oluştur**  
  _Açıklama_: 50 maçlık örnek veri üzerinden tüm pipeline test edilecek.

- [ ] **D2: Value bet işaretleme kontrolü yapılacak**  
  _Açıklama_: Modelin / kuralın doğru şekilde value bet'leri seçip seçmediği doğrulanacak.

---

## 🧬 E. VERİ KALİTESİ VE İZLENEBİLİRLİK

- [ ] **E1: Maçları `match_id` bazında eşsizleştir**  
  _Açıklama_: Aynı maçın farklı timestamp ile iki kez çekilmesi durumları olabilir. `match_id`, `code` veya benzeri bir primary key kullanılarak veri setinde duplicate kontrolü yapılmalı.  
  ✨ Hedef fonksiyon: `deduplicate_matches(df)`

- [ ] **E2: Günlük veri snapshot sistemi kur**  
  _Açıklama_: Her veri çekimi tarih bazlı bir dosya (örn. `oranlar_20250801.pkl`) olarak kaydedilmeli. Böylece geçmiş oran hareketleri izlenebilir, geriye dönük analiz yapılabilir.  
  ✨ Hedef fonksiyon: `save_snapshot(df, date=None)`


🗒 Task güncellemelerinde:
- ✅ ile işaretle
- ✍️ Yeni task'ları varsa altına ekle
