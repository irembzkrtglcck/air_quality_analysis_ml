# Methodology

Bu çalışma, **hava kalitesi** sınıflarını (Good, Moderate, Poor, Hazardous) tahmin etmek için bir uçtan uca makine öğrenmesi hattı (ML pipeline) uygular. Aşağıda veri işleme, modelleme, değerlendirme ve ek analiz adımları özetlenmiştir.

## 1) Veri ve Hedef
- Veri seti: ~**5000 örnek / 9 özellik**.
- Hedef değişken: **Air Quality** (çok sınıflı: Good, Moderate, Poor, Hazardous).
- Sınıf dağılımında dengesizlik bulunduğundan, eğitimde yeniden örnekleme uygulanmıştır.

## 2) Veri Ön İşleme
- **Temizlik:** Fiziksel olarak hatalı değerler (ör. negatif PM/SO₂) temizlendi veya uygun istatistiklerle (medyan) düzeltildi.
- **Ölçekleme:** Sayısal özellikler **StandardScaler** ile normalize edildi.
- **Etiketleme:** Hedef değişken sayısal etiketlere dönüştürüldü (Label Encoding).

## 3) Sınıf Dengesizliği
- Eğitim verisi üzerinde **SMOTE** kullanılarak azınlık sınıfları sentetik örneklerle dengelendi.
- **Test verisi üzerinde SMOTE uygulanmadı** (genelleme/gerçek dünya performansını korumak için).

## 4) Boyut İndirgeme (Analitik)
- **PCA**: Varyansı en iyi temsil eden doğrusal bileşenlerle veri yapısı incelendi (2B/3B görselleştirmeler).
- **LDA**: Sınıf ayrımını maksimize eden doğrultular üzerinden **denetimli** indirgeme.
- **t-SNE**: Doğrusal olmayan komşuluk ilişkilerinin görselleştirilmesi (2B/3B).

> Not: Bu yöntemler **model performansından bağımsız** olarak veri yapısını anlamak ve görselleştirmek için kullanıldı; sınıflandırma modeli eğitiminde ana veri temsili korunmuştur.

## 5) Eğitim / Doğrulama Kurulumu
- **Train/Test bölünmesi:** %80 / %20, **Stratified** (sınıf oranlarını koruyacak şekilde).
- **Çapraz doğrulama:** **5-katlı Stratified K-Fold**.
- **Hiperparametre araması:** Yönetilebilir parametre uzaylarında **GridSearchCV** ile.
- Eğitim, **SMOTE ile dengelenmiş eğitim seti** üzerinde yapıldı; değerlendirme **ham test setinde** raporlandı.

## 6) Modeller
Aşağıdaki sınıflandırıcılar optimize edilip karşılaştırıldı:
- **Logistic Regression** (L2 düzenlileştirme, `C`, `solver` taraması)
- **Decision Tree** (derinlik ve yaprak kısıtları ile aşırı öğrenmenin kontrolü)
- **Random Forest** (ağaç sayısı, derinlik ve bölünme eşikleri)
- **XGBoost** (öğrenme oranı, derinlik, n_estimators, subsample)
- **SVM** (kernel, `C`, `gamma`)
- **KNN** (k, ağırlıklandırma, mesafe metriği)

## 7) Değerlendirme Metrikleri
- **Accuracy**, **Precision**, **Recall**, **F1-Score (macro)**.
- **ROC-AUC (macro, OvR)**.
- **Confusion Matrix** ve **Classification Report** ile hata tiplerinin analizi.

## 8) Sonuçların Özeti (Seçili)
- **Random Forest** test üzerinde **en iyi genel performansı** verdi  
  – Accuracy ≈ **0.949**, F1 (macro) ≈ **0.920**, ROC-AUC (macro) ≈ **0.995**.  
- **XGBoost** çapraz doğrulamada en yüksek skoru elde etti  
  – CV Score ≈ **0.971**, test Accuracy ≈ **0.946**, F1 (macro) ≈ **0.919**, ROC-AUC ≈ **0.993**.  
- Sınıf dengesizliği SMOTE ile iyileştirildi; özellikle **Hazardous** sınıfında kazanım gözlendi.
- Modeller arası farklar düşük olup, ensemble yöntemler (RF, XGB) açık şekilde öne çıktı.

## 9) Denetimsiz Öğrenme (Karşılaştırmalı)
- **K-Means**: Dirsek (elbow) ve Silhouette değerlendirmesiyle **k=4** makul bulundu; sınıflarla kısmi/ölçülebilir örtüşme elde edildi.
- **DBSCAN**: Parametre duyarlı (eps, min_samples); veri yoğunluk yapısından ötürü yüksek gürültü oranı işaretleyebildi.
- Sonuç: Bu veri setinde **K-Means**, DBSCAN’a kıyasla daha **tutarlı** kümeler üretti.

## 10) Tekrarlanabilirlik ve Ortam
- Tohum/rasgelelik: `random_state=42` (uygulandığı yerlerde).
- Ana kütüphaneler: **scikit-learn**, **imbalanced-learn**, **xgboost**, **matplotlib**/**seaborn**.
- Tüm deneyler aynı veri bölünmeleri ve CV protokolüyle yürütülmüştür.

---

### Minimal Çalıştırma Adımları
1. Veriyi yükle, temel temizlik ve **StandardScaler** uygula.  
2. **Train/Test** (%80/%20, stratified) oluştur.  
3. **SMOTE**’u *sadece eğitim setine* uygula.  
4. Modeller için **GridSearchCV + 5-fold** ile hiperparametre tara.  
5. **Test** setinde Accuracy, F1 (macro), ROC-AUC (macro) raporla.  
6. İsteğe bağlı: PCA/LDA/t-SNE ile görselleştir; K-Means/DBSCAN ile kümeleme analizi yap.  
