
# 🛡️ Credit Card Fraud Detection with XGBoost and Feature Engineering

## 📌 Problem Tanımı

Bu projede, IEEE-CIS tarafından sağlanan kredi kartı işlem verileri kullanılarak **kredi kartı dolandırıcılığı (fraud)** tespiti yapılmıştır. Veri seti ciddi şekilde dengesiz olup (~%0.17 fraud), bu nedenle doğru sınıflandırma için veri ön işleme, öznitelik mühendisliği ve güçlü modeller uygulanmıştır.

## 📚 Veri Seti

- Kaynak: [Kaggle - IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- Gözlem sayısı: **1,097,231**
- Özellik sayısı: **472**
- Hedef değişken: `isFraud`

## 🔍 Proje Aşamaları

### 1. EDA (Exploratory Data Analysis)
- Eksik değer analizi
- Sayısal ve kategorik değişken dağılımları
- Fraud ve non-fraud işlemler karşılaştırması

### 2. Feature Engineering
- `TransactionDT` üzerinden gün/saat çıkarımı yapıldı
- E-posta domain özetleme (`e-mail_cols`) oluşturuldu
- Kart bilgileri üzerinden kategorileştirme (`card_cols`) yapıldı
- Yeni oluşturulan bazı değişkenler: `ProductCD_W`, `new_card6_transaction_mean`

### 3. Encoding
- Kategorik değişkenlere **One-Hot Encoding** ve **Frequency Encoding** uygulandı
- Memory optimization yapıldı

### 4. Modelleme
- **Random Forest**, **GBM**, **LightGBM** ve **XGBoost** modelleri denendi
- En iyi performansı gösteren **XGBoost** ile devam edildi
- Model değerlendirme ROC-AUC skorlarına göre yapıldı:
  - Random Forest: **0.7256**
  - GBM: **0.8266**
  - LightGBM: **0.8234**
  - XGBoost: **0.8334**

### 5. Hiperparametre Optimizasyonu Sonrası Performans Artışı
- Test seti **Recall** değeri: 0.50 → **0.57**
- Cross Validation **ROC-AUC** skoru: 0.9424 → **0.9599**
- **F1 Skoru**: 0.65 → **0.71**
- Olasılıklar üzerinden hesaplanan **ROC-AUC** skoru: 0.948 → **0.967**
- Önem kazanan değişkenler: `ProductCD_W`, `new_card6_transaction_mean`

## 🧰 Kullanılan Teknolojiler

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- XGBoost
- LightGBM

## 📁 Proje Dosya Yapısı

```bash
├── 01_eda.py
├── 02_feature_engineering.py
├── 03_encoding.py
├── 04_modelling.py
├── README.md
├── requirements.txt
```

## 📌 Notlar
Bu proje, Miuul Data Scientist Bootcamp kapsamında bitirme projesi olarak geliştirilmiştir. Model seçimi, hiperparametre optimizasyonu ve veri ön işleme süreçleri özenle ele alınmıştır.
