# 📊 Veri Seti Açıklaması

## Genel Bilgiler

**Kaynak:** [Kaggle - Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)

**Dosya Adı:** `updated_pollution_dataset.csv`

**Boyut:**
- **Örnek Sayısı:** 5,000
- **Özellik Sayısı:** 9 (bağımsız değişken)
- **Hedef Değişken:** 1 (Air Quality)

---

## 📋 Değişken Açıklamaları

| Sütun Adı | Tip | Açıklama | Birim |
|-----------|-----|----------|-------|
| **Temperature** | Sayısal | Ortam sıcaklığı | °C |
| **Humidity** | Sayısal | Bağıl nem oranı | % |
| **PM2.5** | Sayısal | 2.5 mikrometreden küçük partikül madde | µg/m³ |
| **PM10** | Sayısal | 10 mikrometreden küçük partikül madde | µg/m³ |
| **NO2** | Sayısal | Azot dioksit konsantrasyonu | ppb |
| **SO2** | Sayısal | Kükürt dioksit konsantrasyonu | ppb |
| **CO** | Sayısal | Karbon monoksit konsantrasyonu | ppm |
| **Proximity_to_Industrial_Areas** | Sayısal | Sanayi bölgelerine yakınlık skoru | 0-10 |
| **Population_Density** | Sayısal | Nüfus yoğunluğu | kişi/km² |
| **Air Quality** | Kategorik | Hava kalitesi sınıfı (hedef) | Good/Moderate/Poor/Hazardous |

---

## 🎯 Hedef Değişken Dağılımı

| Sınıf | Örnek Sayısı | Oran |
|-------|--------------|------|
| **Good** | 2,000 | 40% |
| **Moderate** | 1,500 | 30% |
| **Poor** | 1,000 | 20% |
| **Hazardous** | 500 | 10% |

**Not:** Veri setinde sınıf dengesizliği bulunmaktadır. SMOTE yöntemi ile eğitim verisi dengelenmiştir.

---

## 🔧 Veri Kalitesi Sorunları ve Çözümler

### ❌ Tespit Edilen Problemler

1. **Negatif Değerler:**
   - PM10: 1 adet negatif değer
   - SO2: 30 adet negatif değer

2. **Aykırı Değerler (Outliers):**
   - PM2.5, PM10, NO2, SO2, CO değişkenlerinde kutu grafiklerinde belirgin aykırı değerler

### ✅ Uygulanan Çözümler

1. **Negatif Değer Temizleme:**
   ```python
   # Fiziksel olarak negatif olamayan değişkenler için medyan ile doldurma
   air_quality.loc[air_quality['PM10'] < 0, 'PM10'] = air_quality['PM10'].median()
   air_quality.loc[air_quality['SO2'] < 0, 'SO2'] = air_quality['SO2'].median()
   ```

2. **Sınıf Dengesizliği:**
   ```python
   # SMOTE (Synthetic Minority Over-sampling Technique)
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   ```

3. **Standardizasyon:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

---

## 📥 Veri İndirme

### Manuel İndirme
1. [Kaggle veri seti sayfasına](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) gidin
2. "Download" butonuna tıklayın
3. İndirilen ZIP dosyasını `data/` klasörüne çıkarın

### Python ile Otomatik İndirme
```python
import kagglehub

# Kaggle API credentials gereklidir
path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
print(f"Dataset downloaded to: {path}")
```

**Not:** Kaggle API kullanımı için `kaggle.json` dosyanız hazır olmalıdır.

---

## 📊 Temel İstatistikler

```
Temperature:     min=-10.0°C,  max=45.0°C,   mean=17.5°C
Humidity:        min=15.0%,    max=95.0%,    mean=55.3%
PM2.5:           min=0.0,      max=250.0,    mean=75.4 µg/m³
PM10:            min=0.0,      max=400.0,    mean=120.2 µg/m³
NO2:             min=0.0,      max=100.0,    mean=35.6 ppb
SO2:             min=0.0,      max=80.0,     mean=28.3 ppb
CO:              min=0.0,      max=10.0,     mean=3.2 ppm
```

---

## 🔗 İlgili Bağlantılar

- **WHO Hava Kalitesi Standartları:** https://www.who.int/air-pollution/guidelines/en/
- **EPA Air Quality Index:** https://www.airnow.gov/aqi/aqi-basics/
- **Kaggle Notebook'lar:** https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment/code

---

## ⚠️ Dikkat

Bu veri seti eğitim ve araştırma amaçlıdır. Gerçek zamanlı hava kalitesi tahminleri için güncel sensör verileri kullanılmalıdır.