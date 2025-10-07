#  Air Quality Analysis with Machine Learning

Hava kalitesi verilerinin makine öğrenmesi algoritmaları ile analizi ve sınıflandırılması projesi.

##  Proje Hakkında

Bu proje, hava kalitesi göstergelerini (PM2.5, PM10, NO2, SO2, CO vb.) kullanarak hava kalitesini sınıflandıran kapsamlı bir makine öğrenmesi çalışmasıdır.

##  Amaç

- Hava kalitesi verilerinin keşifsel veri analizi (EDA)
- Boyut indirgeme teknikleri (PCA, LDA, t-SNE)
- Çoklu sınıflandırma algoritmaları ile modelleme
- En iyi performans gösteren modelin belirlenmesi

##  Veri Seti

- **Kaynak:** [Kaggle - Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)
- **Özellikler:** PM2.5, PM10, NO2, SO2, CO, Ozone, Temperature, Humidity, Wind Speed
- **Hedef Değişken:** Air Quality (Good, Moderate, Poor, Hazardous)

##  Kullanılan Teknolojiler

- **Python 3.x**
- **Kütüphaneler:**
  - pandas, numpy (Veri işleme)
  - matplotlib, seaborn (Görselleştirme)
  - scikit-learn (Makine öğrenmesi)
  - xgboost (Gradient boosting)
  - imblearn (SMOTE - Sınıf dengeleme)

##  Metodoloji

### 1. Veri Ön İşleme
- Eksik değer analizi
- Negatif değerlerin temizlenmesi (median ile doldurma)
- Özellik ölçeklendirme (StandardScaler)

### 2. Keşifsel Veri Analizi (EDA)
- Betimleyici istatistikler
- Korelasyon analizi (Pearson & Spearman)
- Boxplot ile aykırı değer tespiti
- Hedef değişken dağılımı

### 3. Boyut İndirgeme
- **PCA** (Principal Component Analysis) - 2B ve 3B
- **LDA** (Linear Discriminant Analysis) - 2B ve 3B
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding) - 2B ve 3B

### 4. Sınıf Dengeleme
- **SMOTE** (Synthetic Minority Over-sampling Technique)

### 5. Makine Öğrenmesi Modelleri

#### Kullanılan Algoritmalar:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Gradient Boosting
7. XGBoost

#### Hiperparametre Optimizasyonu:
- GridSearchCV ile en iyi parametreler bulundu
- 5-fold Stratified Cross-Validation

### 6. Model Değerlendirme Metrikleri
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- ROC AUC (One-vs-Rest)
- Confusion Matrix

### 7. Kümeleme Analizi
- **K-Means:** Elbow method ve Silhouette score ile optimal k belirleme
- **DBSCAN:** k-distance grafiği ile epsilon optimizasyonu

## 📈 Sonuçlar

| Model | CV Score | Accuracy | F1 (macro) | ROC AUC |
|-------|----------|----------|------------|---------|
| Random Forest | 0.958 | **0.949** | **0.920** | **0.995** |
| XGBoost | **0.971** | 0.946 | 0.919 | 0.993 |
| SVM | 0.949 | 0.939 | 0.913 | 0.992 |
| Logistic Regression | 0.943 | 0.934 | 0.905 | 0.989 |
| Decision Tree | 0.923 | 0.928 | 0.892 | 0.933 |
| KNN | 0.925 | 0.921 | 0.881 | 0.985 |

**En İyi Model:** Random Forest  
**Test Accuracy:** 0.949 (94.9%)  
**F1-Score (macro):** 0.920  
**ROC AUC:** 0.995

*Not: XGBoost en yüksek CV Score'u elde etse de (0.971), Random Forest test setinde daha iyi genelleme performansı göstermiştir.*

### Öne Çıkan Bulgular:
- Ensemble metodları (RF, GB, XGBoost) en yüksek performansı gösterdi
- SMOTE uygulaması minority sınıfların performansını artırdı
- PM2.5 ve PM10 en önemli özellikler olarak belirlendi

##  Görselleştirmeler

Proje şunları içerir:
- Korelasyon matrisleri (heatmap)
- Boxplot grafikleri
- PCA/LDA/t-SNE scatter plotları (2B ve 3B)
- Confusion matrix'ler
- ROC eğrileri (sınıf bazlı ve macro-average)
- Feature importance grafikleri
- Model karşılaştırma grafikleri
- Kümeleme görselleştirmeleri

##  Kullanım

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Çalıştırma

```bash
python air_quality_analysis.py
```

##  Dosya Yapısı

```
air_quality_analysis_ml/
│
├── air_quality_analysis.py    # Ana analiz scripti
├── requirements.txt            # Gerekli kütüphaneler
├── README.md                   # Proje dokümantasyonu
└── results/                    # Sonuç görselleri (opsiyonel)
```

##  Geliştirme Önerileri

- [ ] Deep Learning modelleri (LSTM, CNN) ekleme
- [ ] Zaman serisi analizi
- [ ] Web tabanlı dashboard (Streamlit/Dash)
- [ ] Model deployment (Flask API)
- [ ] Gerçek zamanlı veri entegrasyonu

##  Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

##  Lisans

Bu proje [MIT License](LICENSE) altında lisanslanmıştır.

##  İletişim

- **GitHub:** [@irembzkrtglcck](https://github.com/irembzkrtglcck)
- **Linkedin:** [linkedin.com/in/irembozkurt7/](https://www.linkedin.com/in/irembozkurt7/)
- **Proje Linki:** [https://github.com/irembzkrtglcck/air_quality_analysis_ml](https://github.com/irembzkrtglcck/air_quality_analysis_ml)

## 🙏 Teşekkürler

- Kaggle - Veri seti için
- Scikit-learn - Makine öğrenmesi araçları için
- Anthropic Claude - Kod geliştirme desteği için

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
