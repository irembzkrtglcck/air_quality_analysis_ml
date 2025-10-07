# 🌍 Hava Kalitesi Veri Kümesinin Makine Öğrenmesi ile Analizi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Hava kalitesi verilerinin 6 farklı makine öğrenmesi algoritması ile sınıflandırılması, boyut indirgeme teknikleri ve kümeleme analizlerinin kapsamlı karşılaştırılması.

## 📊 Proje Özeti

Bu çalışmada, 5000 örnek ve 9 özellik içeren hava kalitesi veri seti üzerinde:
- ✅ **Sınıflandırma:** 6 farklı ML algoritması (LR, DT, RF, XGBoost, SVM, KNN)
- ✅ **Optimizasyon:** GridSearchCV ile hiperparametre ayarı
- ✅ **Dengeleme:** SMOTE ile sınıf dengesizliği çözümü
- ✅ **Boyut İndirgeme:** PCA, LDA, t-SNE karşılaştırması
- ✅ **Kümeleme:** K-Means ve DBSCAN analizi

### 🎯 Hedef Sınıflar
| Sınıf | Açıklama | Dağılım |
|-------|----------|---------|
| **Good** | İyi hava kalitesi | %40 |
| **Moderate** | Orta derece | %30 |
| **Poor** | Kötü | %20 |
| **Hazardous** | Tehlikeli | %10 |

## 🔬 Model Performans Karşılaştırması

| Model | CV Score | Test Accuracy | F1 Score | ROC AUC |
|-------|----------|---------------|----------|---------|
| **Random Forest** 🥇 | 0.958 | **0.949** | **0.920** | **0.995** |
| **XGBoost** 🥈 | **0.971** | 0.946 | 0.919 | 0.993 |
| **SVM** 🥉 | 0.949 | 0.939 | 0.913 | 0.992 |
| Logistic Regression | 0.943 | 0.934 | 0.905 | 0.989 |
| KNN | 0.925 | 0.921 | 0.881 | 0.985 |
| Decision Tree | 0.923 | 0.928 | 0.892 | 0.933 |

### 📈 Önemli Özellikler (Feature Importance)
1. **PM2.5** → 0.28 (En önemli)
2. **PM10** → 0.24
3. **CO** → 0.18
4. **NO₂** → 0.12
5. **SO₂** → 0.10

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- pip paket yöneticisi

### Adımlar

```bash
# 1. Repo'yu klonla
git clone https://github.com/irembzkrtglcck/air_quality_analysis_ml.git
cd air_quality_analysis_ml


# 2. Sanal ortam oluştur (önerilen)
python -m venv venv

# Aktivasyon (Windows)
venv\Scripts\activate

# Aktivasyon (Mac/Linux)
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

## 📚 Kullanım

### Jupyter Notebook ile
```bash
jupyter notebook notebooks/air_quality_analysis.ipynb
```

### Python Script ile
```bash
python src/analysis.py
```

### Google Colab'da Çalıştırma
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULLANICIADIN/air-quality-ml-analysis/blob/main/notebooks/air_quality_analysis.ipynb)

## 📂 Proje Yapısı

```
air-quality-ml-analysis/
│
├── data/                      # Veri dosyaları
│   └── README.md             # Veri seti açıklaması
│
├── notebooks/                 # Jupyter notebook'lar
│   └── air_quality_analysis.ipynb
│
├── src/                       # Python kaynak kodları
│   └── analysis.py
│
├── reports/                   # Raporlar ve görseller
│   ├── Hava_Kalitesi_Analizi.pdf
│   └── figures/
│
├── docs/                      # Detaylı dokümantasyon
│   └── METHODOLOGY.md
│
├── README.md                  # Bu dosya
├── requirements.txt           # Python bağımlılıkları
├── .gitignore
└── LICENSE
```

## 🎓 Metodoloji

### 1. Veri Ön İşleme
- ❌ **Negatif Değer Temizleme:** PM10 (1 adet), SO₂ (30 adet) → medyan ile doldurma
- ⚖️ **Sınıf Dengeleme:** SMOTE uygulaması (eğitim setinde)
- 📏 **Standardizasyon:** StandardScaler ile normalizasyon

### 2. Model Eğitimi
- 🔍 **Hiperparametre Optimizasyonu:** GridSearchCV
- 📊 **Çapraz Doğrulama:** 5-katlı StratifiedKFold
- 📉 **Train/Test Ayrımı:** %80 / %20

### 3. Boyut İndirgeme
- **PCA (5 bileşen):** Accuracy 0.921 (▼2.8%)
- **LDA (3 bileşen):** Accuracy 0.935 (▼1.4%)
- **Sonuç:** Orijinal 9 özellik en iyi performansı sağladı

### 4. Kümeleme Analizi
- **K-Means (k=4):** Gerçek etiketlerle %68 örtüşme
- **DBSCAN (eps=0.7):** En dengeli parametre

## 📈 Ana Bulgular

### ✨ Başarılar
1. **Rastgele Orman** en dengeli model (accuracy, F1, ROC AUC'de üstün)
2. **XGBoost** en yüksek CV score (0.971) ancak test setinde RF'nin gerisinde
3. **SMOTE** uygulaması Hazardous sınıfında recall'u %62'den %88'e çıkardı
4. **PM2.5 ve PM10** hava kalitesi tahmini için en kritik özellikler
5. GridSearchCV ile **%3-5 performans artışı** sağlandı

### ⚠️ Zorluklar
- **Moderate-Poor sınıfları** arasında geçişkenlik (confusion matrix'te belirgin)
- **DBSCAN** parametre hassasiyeti yüksek (eps 0.5→1.3 arası %8-58 gürültü)
- **Boyut indirgeme** bu veri setinde performans kaybına neden oldu

## 📊 Görselleştirmeler

Proje içerisinde yer alan ana görseller:
- 🔥 Confusion Matrix'ler (6 model)
- 📊 ROC Eğrileri (Multi-class)
- 🎨 PCA/LDA/t-SNE görselleştirmeleri
- 📈 Feature Importance grafikleri
- 🔵 K-Means kümeleme sonuçları

## 📖 Dokümantasyon

Detaylı metodoloji ve sonuçlar için:
- 📄 [Tam Rapor (PDF)](reports/Hava_Kalitesi_Analizi.pdf)
- 📝 [Metodoloji Detayları](docs/METHODOLOGY.md)
- 📊 [Veri Seti Açıklaması](data/README.md)

## 🔗 Veri Seti

**Kaynak:** [Kaggle - Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)

**Özellikler:**
- Temperature (°C)
- Humidity (%)
- PM2.5 (µg/m³)
- PM10 (µg/m³)
- NO₂, SO₂, CO
- Proximity to Industrial Areas
- Population Density

## 🚀 Gelecek Geliştirmeler

- [ ] LSTM/GRU ile zaman serisi tahmini
- [ ] SHAP values ile model açıklanabilirliği
- [ ] LightGBM, CatBoost eklenmesi
- [ ] Ensemble voting (RF + XGB + SVM)
- [ ] Web API deployment (Flask/FastAPI)
- [ ] Real-time prediction dashboard

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:
1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/YeniOzellik`)
5. Pull Request açın

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👤 Yazar

**İrem BOZKURT GÜLÇİÇEK**  
📧 Email: irem.bozkurt@example.com  
🔗 LinkedIn: [linkedin.com/in/irembozkurt](https://linkedin.com/in/irembozkurt)

### 🎓 Danışman
**Dr. Öğr. Üyesi Süha TUNA**  
Proje Danışmanı

## 🙏 Teşekkürler

- Kaggle topluluğu (veri seti için)
- scikit-learn katkıda bulunanlar
- XGBoost geliştirici ekibi

---

⭐ **Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

📅 **Son Güncelleme:** Ekim 2025