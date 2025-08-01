# 🎯 Bilyoner Value Bet Projesi - Task Listesi

Bu dosya, oran tahmini projesinde yapılacak tüm işleri kategorize ederek takip etmek için hazırlanmıştır.

---

## ✅ A. VERİ TEMİZLİĞİ & HAZIRLIK

- [x] **A1: Tüm skorları sil ve yeniden kaydet**  
  ✨ Fonksiyon: `refresh_scores()`

- [x] **A2: MS oranı olmayan maçları sil**  
  ✨ Fonksiyon: `filter_missing_ms()`

- [x] **A3–A5: Fazlalık sütunların tespiti ve temizliği**  
  ✨ Fonksiyonlar:
    - `detect_useless_columns(df)`
    - `nullify_empty_markets(df, useless_df)`
    - `drop_useless_columns(df, useless_df)`

---

## 🚀 B. PERFORMANS OPTİMİZASYONU

- [ ] **B1: Insert işlemlerini hızlandır**  
  _CSV yavaşsa → Parquet / Feather veya `to_sql` bulk yöntemleri değerlendirilecek_  
  ✨ Fonksiyon: `save_data(df, method="auto")`

- [ ] **B2: Lambda ve apply kullanımlarını vektörel hale getir**  
  _apply yerine NumPy vektörleri, `np.where` gibi yapılar kullanılacak_  
  ✨ Fonksiyon: `vectorize_processing()`

- [ ] **B3: Zamanlama loglaması eklensin**  
  _Her fonksiyonun çalışma süresi ölçülecek_  
  ✨ Fonksiyon: `@timing_logger` veya `with Timer()`

---

## 🧱 C. YAPISAL DÜZENLEME & MODÜLERLİK

- [ ] **C1: Kodları `.py` dosyalarına böl (scraper.py, processor.py, predictor.py)**  
  _Notebook'lar sadece test/görselleştirme için kullanılacak_

- [ ] **C2: Ortak işlemleri `utils.py` dosyasına taşı**  
  _Tarihli snapshot, safe load/save, timer gibi işlemler için yardımcı fonksiyonlar yazılacak_

- [ ] **C3: Value bet analiz fonksiyonlarını yaz**  
  _Oranlara göre beklenen değer (`EV`) ve implied probability hesaplayan yapılar kurulacak_  
  ✨ Örn: `calculate_ev(prob, odds)`

---

## 🧪 D. TEST & DOĞRULAMA

- [ ] **D1: Test dataset'i oluştur**  
  _50 maçlık veriyle hızlı test yapılacak, pipeline doğrulanacak_

- [ ] **D2: Value bet kararlarını doğrula**  
  _Model/hesaplama mantıklı value bet'leri doğru işaretliyor mu_

---

## 🧬 E. VERİ KALİTESİ & TAKİP

- [ ] **E1: `match_id` bazlı unique kontrol**  
  _Duplicate maçlar varsa temizlenecek_  
  ✨ Fonksiyon: `deduplicate_matches(df)`

- [ ] **E2: Günlük snapshot sistematiği kur**  
  _Tüm çıktılar tarihli versiyon olarak kaydedilecek_  
  ✨ Fonksiyon: `save_snapshot(df, date=None)`

---

🗒 Task güncellemelerinde:
- [x] ile tamamladıklarını işaretle
- ✍️ Yeni task'ları gerektiğinde altına ekle
