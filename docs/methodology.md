# 🔬 Metodoloji Dokümantasyonu

Bu belge, hava kalitesi analizi projesinde kullanılan tüm yöntemlerin detaylı açıklamalarını içerir.

---

## 📚 İçindekiler

1. [Veri Ön İşleme](#1-veri-ön-işleme)
2. [Özellik Mühendisliği](#2-özellik-mühendisliği)
3. [Boyut İndirgeme Teknikleri](#3-boyut-indirgeme-teknikleri)
4. [Sınıflandırma Modelleri](#4-sınıflandırma-modelleri)
5. [Kümeleme Analizi](#5-kümeleme-analizi)
6. [Değerlendirme Metrikleri](#6-değerlendirme-metrikleri)

---

## 1. Veri Ön İşleme

### 1.1. Eksik Değer Kontrolü

```python
# Eksik değer kontrolü
print(air_quality.isnull().sum())
```

**Sonuç:** Veri setinde eksik değer bulunmamaktadır.

### 1.2. Negatif Değer Temizleme

**Problem:** Fiziksel olarak negatif olamayan değişkenlerde negatif değerler tespit edildi.

| Değişken | Negatif Sayısı |
|----------|----------------|
| PM10 | 1 |
| SO2 | 30 |

**Çözüm:** Medyan imputation

```python
# Medyan ile doldurma
air_quality.loc[air_quality['PM10'] < 0, 'PM10'] = air_quality['PM10'].median()
air_quality.loc[air_quality['SO2'] < 0, 'SO2'] = air_quality['SO2'].median()
```

**Neden Medyan?**
- Aykırı değerlerden etkilenmez
- Dağılımı korur
- Ortalamaya göre daha robust

### 1.3. Standardizasyon

**Yöntem:** StandardScaler (Z-score normalizasyonu)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Formül:**
```
z = (x - μ) / σ
```

**Neden StandardScaler?**
- Mesafe tabanlı algoritmalar için kritik (KNN, SVM)
- PCA için gerekli
- Farklı ölçekli değişkenleri aynı düzeye getirir

---

## 2. Özellik Mühendisliği

### 2.1. Label Encoding (Hedef Değişken)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Eşleşmeler:
# Good → 0
# Hazardous → 1
# Moderate → 2
# Poor → 3
```

### 2.2. Sınıf Dengesizliği - SMOTE

**Problem:** Dengesiz sınıf dağılımı

| Sınıf | SMOTE Öncesi | SMOTE Sonrası |
|-------|--------------|---------------|
| Good | 1,600 | 1,600 |
| Moderate | 1,200 | 1,600 |
| Poor | 800 | 1,600 |
| Hazardous | 400 | 1,600 |

**SMOTE Algoritması:**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Nasıl Çalışır?**
1. Azınlık sınıfından bir örnek seç
2. K-en yakın komşularını bul
3. Komşularla doğrusal interpolasyon yap
4. Sentetik örnek oluştur

**Avantajları:**
- Overfitting riski düşük
- Test setine dokunmaz
- Hassas sınıflarda performansı artırır

---

## 3. Boyut İndirgeme Teknikleri

### 3.1. PCA (Principal Component Analysis)

**Amaç:** Varyansı maksimize eden yeni eksenler bul

**Uygulama:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Sonuçlar:**
- İlk 2 bileşen: %70.69 varyans
- İlk 3 bileşen: %77.3 varyans

**Avantajları:**
- Unsupervised (etiket gerektirmez)
- Çoklu bağlantıyı (multicollinearity) azaltır

**Dezavantajları:**
- Yorumlanabilirlik kaybı
- Sınıf bilgisini kullanmaz

### 3.2. LDA (Linear Discriminant Analysis)

**Amaç:** Sınıflar arası ayrımı maksimize et

**Formül:**
```
maximize: (between-class variance) / (within-class variance)
```

**Uygulama:**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
```

**Kısıt:** n_components ≤ (n_classes - 1)

**Avantajları:**
- Sınıflandırma için optimize edilmiş
- PCA'dan daha iyi sınıf ayrımı

**Dezavantajları:**
- Supervised (etiket gerektirir)
- Doğrusallık varsayımı

### 3.3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Amaç:** Yüksek boyutlu komşulukları düşük boyutta koru

**Uygulama:**
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)
```

**Parametreler:**
- **perplexity:** Komşu sayısı (5-50 arası önerilir)
- **learning_rate:** Optimizasyon hızı

**Avantajları:**
- Doğrusal olmayan ilişkileri yakalar
- Görselleştirmede mükemmel

**Dezavantajları:**