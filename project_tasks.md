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

## 🚀 B. VERİ AKIŞI & PERFORMANS OPTİMİZASYONU

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

## 📥 C. VERİ TOPLAMA & GÜNCELLEME

- [ ] **C1: Yeni maçları scrape et ve CSV'ye ekle**  
  _Günlük yeni maçlar otomatik scrape edilerek mevcut veri setine eklenmeli_  
  ✨ Fonksiyon: `append_new_matches()` (önerilen)

- [ ] **C2: Günlük snapshot sistematiği kur**  
  _Tüm çıktılar tarihli versiyon olarak kaydedilmeli_  
  ✨ Fonksiyon: `save_snapshot(df, date=None)`

---

## 📐 D. MODELLEME & ANALİZ FONKSİYONLARI

- [x] **D1: Value bet analiz fonksiyonlarını yaz**  
  ✨ Fonksiyonlar: `calculate_ev(prob, odds)`, `implied_probability(odds)`, `run_value_analysis(...)`

- [ ] **D2: Bütün bahis tipleri için EV/olasılık hesabı kur**  
  _Sadece MS değil; Çifte Şans, Alt/Üst vb. için de value analiz yapısı kurulacak_  
  ✨ Fonksiyon: `run_value_analysis_by_market(...)`

- [ ] **D3: Lambda apply yerine vektörel analiz fonksiyonu**  
  _Tahmin ve analizler vektörel hale getirilecek_

---

## 🧱 E. YAPISAL DÜZENLEME & MODÜLERLİK

- [ ] **E1: Kodları `.py` dosyalarına böl (scraper.py, processor.py, predictor.py, etc.)**  
  _Notebook'lar sadece test/görselleştirme için olacak_

- [ ] **E2: Ortak işlemleri `utils.py` dosyasına taşı**  
  _Tarihli snapshot, safe load/save, timer gibi işlemler için yardımcı fonksiyonlar yazılacak_

---

## 🧪 F. TEST & DOĞRULAMA

- [ ] **F1: Test dataset'i oluştur**  
  _50 maçlık veriyle hızlı test yapılacak, pipeline doğrulanacak_

- [ ] **F2: Value bet kararlarını doğrula (backtest)**  
  _Model mantıklı value bet'leri doğru işaretliyor mu_

- [ ] **F3: Tüm maçlar için batch tahmin ve rapor**  
  _Backtest'te tüm maçlara value bet etiketi eklenerek sonuçlar csv'ye kaydedilecek_

---

## 🧬 G. VERİ KALİTESİ VE İZLENEBİLİRLİK

- [ ] **G1: `match_id` bazlı duplicate kontrolü**  
  _Aynı maç birden fazla kez varsa temizlenecek_  
  ✨ Fonksiyon: `deduplicate_matches(df)`

---

🗒 Güncellemelerde:
- [x] ile tamamladıklarını işaretle
- ✍️ Yeni task'ları gerektiğinde altına ekle
