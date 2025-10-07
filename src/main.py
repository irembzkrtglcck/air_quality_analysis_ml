import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import kagglehub, os

# =============================
# VERİYİ İNDİR & YÜKLE
# =============================
path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
air_quality = pd.read_csv(path + "/updated_pollution_dataset.csv")

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

print("▶ Veri boyutu:", air_quality.shape)

print("\n▶ Veri tipi bilgisi:")
air_quality.info()

print("\n▶ İlk 5 gözlem:")
print(air_quality.head())

print("\n▶ Eksik değer sayısı (sütun bazında):")
print(air_quality.isnull().sum())

TARGET_COL = "Air Quality"

# =============================
# KATEGORİK & SAYISAL SÜTUNLAR
# =============================
cat_cols = air_quality.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET_COL]

if len(cat_cols) == 0:
    print("Veri setinde (target hariç) kategorik değişken bulunmamaktadır.")
else:
    for col in cat_cols:
        print(f"\n--- {col} ---")
        print("Benzersiz değer sayısı:", air_quality[col].nunique())
        print("Benzersiz değerler:", air_quality[col].unique())
        print("\nDeğer frekansları:")
        print(air_quality[col].value_counts())

num_cols = air_quality.select_dtypes(include=["number"]).columns.tolist()

print("▶ Sayısal değişkenlerin betimleyici istatistikleri:")
desc = air_quality[num_cols].describe().T
print(desc)

print("\n▶ Medyan (50%):")
print(air_quality[num_cols].median())

# =============================
# NEGATİF DEĞER KONTROL & TEMİZLEME
# =============================
cols_physical_nonneg = [c for c in ["PM2.5","PM10","NO2","SO2","CO"] if c in air_quality.columns]
if cols_physical_nonneg:
    print("\n▶ Negatif değer kontrolü (fiziken pozitif olması beklenen sütunlar):")
    for c in cols_physical_nonneg:
        neg_cnt = (air_quality[c] < 0).sum()
        if neg_cnt > 0:
            print(f"  - {c}: {neg_cnt} adet negatif değer VAR")
        else:
            print(f"  - {c}: negatif değer YOK")

analysis_cols = [c for c in num_cols if c != TARGET_COL]
print(f"Analizdeki sayısal sütunlar ({len(analysis_cols)}): {analysis_cols}")

print("▶ Veri temizleme öncesi:")
if 'PM10' in air_quality.columns:
    print(f"PM10 negatif değer: {(air_quality['PM10'] < 0).sum()}")
if 'SO2' in air_quality.columns:
    print(f"SO2 negatif değer: {(air_quality['SO2'] < 0).sum()}")

if 'PM10' in air_quality.columns:
    air_quality.loc[air_quality['PM10'] < 0, 'PM10'] = air_quality['PM10'].median()
if 'SO2' in air_quality.columns:
    air_quality.loc[air_quality['SO2'] < 0, 'SO2'] = air_quality['SO2'].median()

print("\n▶ Veri temizleme sonrası:")
if 'PM10' in air_quality.columns:
    print(f"PM10 negatif değer: {(air_quality['PM10'] < 0).sum()}")
if 'SO2' in air_quality.columns:
    print(f"SO2 negatif değer: {(air_quality['SO2'] < 0).sum()}")

# =============================
# KORELASYONLAR
# =============================
pearson_corr = air_quality[analysis_cols].corr(method="pearson")
plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Pearson Korelasyon Matrisi")
plt.tight_layout()
plt.show()

spearman_corr = air_quality[analysis_cols].corr(method="spearman")
plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Spearman Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# Pearson-Spearman farkları
show_diff = True
if show_diff:
    diff = (pearson_corr - spearman_corr).abs()
    plt.figure(figsize=(10, 7))
    mask = np.triu(np.ones_like(diff, dtype=bool))
    sns.heatmap(diff, mask=mask, annot=True, fmt=".2f",
                cmap="Reds", square=True, cbar_kws={"shrink": .8})
    plt.title("Pearson - Spearman Mutlak Farkları")
    plt.tight_layout()
    plt.show()

# En yüksek korelasyon özet tablosu (Pearson)
_top_k = 8
_thr = 0.50
pairs = []
for i in range(len(analysis_cols)):
    for j in range(i+1, len(analysis_cols)):
        r = pearson_corr.iloc[i, j]
        if abs(r) >= _thr:
            pairs.append((analysis_cols[i], analysis_cols[j], r))

pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:_top_k]
if pairs_sorted:
    summary_df = pd.DataFrame(pairs_sorted, columns=["Değişken1", "Değişken2", "Pearson r"])
    print(summary_df.round(3))
else:
    print(f"|r| >= {_thr} eşik üzerinde çift bulunamadı.")

# =============================
# BOXPLOT'LAR
# =============================
plt.figure(figsize=(15, 8))
air_quality[num_cols].boxplot()
plt.title("Sayısal Değişkenler için Kutu Grafikleri (Boxplot)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45)
plt.show()

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=air_quality[col])
    plt.title(f"{col} için Kutu Grafiği", fontsize=12, fontweight="bold")
    plt.show()

# =============================
# HEDEF DEĞİŞKEN DAĞILIMI
# =============================
print("Hedef değişken değer frekansları:")
freq = air_quality[TARGET_COL].value_counts()
print(freq)

print("\nHedef değişken oranları (%):")
ratio = air_quality[TARGET_COL].value_counts(normalize=True) * 100
print(ratio.round(2))

plt.figure(figsize=(6,4))
sns.countplot(x=air_quality[TARGET_COL], hue=air_quality[TARGET_COL], palette="Set2", legend=False)
plt.title("Air Quality Hedef Değişkeninin Dağılımı", fontsize=14, fontweight="bold")
plt.xlabel("Air Quality Sınıfları")
plt.ylabel("Frekans")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.pie(freq, labels=freq.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set2"))
plt.title("Air Quality Sınıflarının Dağılımı (%)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# =============================
# ENCODING & STANDARDIZATION
# =============================
from sklearn.preprocessing import LabelEncoder, StandardScaler

X = air_quality.drop(columns=[TARGET_COL])
y = air_quality[TARGET_COL]

le = LabelEncoder()
y_enc = le.fit_transform(y)
label_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
print("Hedef sınıf-etiket eşleşmeleri:", label_mapping)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(air_quality[num_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=num_cols)
print("Dönüşüm sonrası şekil:", X_scaled_df.shape)
print(X_scaled_df.head())

# =============================
# BOYUT İNDİRGEME: PCA / LDA / t-SNE
# =============================
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# PCA (2B)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_df)
print("PCA bileşen varyans oranları:", np.round(pca.explained_variance_ratio_, 4))
print("Toplam açıklanan varyans:", np.round(pca.explained_variance_ratio_.sum(), 4))

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_enc, alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (2B) - İndirgenmiş Uzay")
plt.colorbar(label="Air Quality (etiket)")
plt.tight_layout(); plt.show()

# LDA (<= 2B)
n_classes = len(np.unique(y_enc))
n_comp = min(2, n_classes - 1)
lda = LDA(n_components=n_comp)
X_lda = lda.fit_transform(X_scaled_df, y_enc)

plt.figure(figsize=(8,6))
if n_comp == 1:
    plt.scatter(X_lda[:,0], np.zeros_like(X_lda[:,0]), c=y_enc, alpha=0.7)
    plt.ylabel("(sabit)"); plt.xlabel("LD1")
else:
    plt.scatter(X_lda[:,0], X_lda[:,1], c=y_enc, alpha=0.7)
    plt.xlabel("LD1"); plt.ylabel("LD2")
plt.title(f"LDA ({n_comp}B) - Sınıf Ayrımı Odaklı İndirgeme")
plt.colorbar(label="Air Quality (etiket)")
plt.tight_layout(); plt.show()

# t-SNE (2B)
tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled_df)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_enc, alpha=0.7)
plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2"); plt.title("t-SNE (2B) - Doğrusal Olmayan İndirgeme")
plt.colorbar(label="Air Quality (etiket)")
plt.tight_layout(); plt.show()

# 3B PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca3 = pca_3d.fit_transform(X_scaled_df)
print("PCA (3B) varyans oranları:", np.round(pca_3d.explained_variance_ratio_, 4))
print("Toplam açıklanan varyans:", np.round(pca_3d.explained_variance_ratio_.sum(), 4))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=y_enc, alpha=0.7)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title("PCA (3B) - İndirgenmiş Uzay")
plt.colorbar(sc, label="Air Quality (etiket)")
plt.show()

# 3B LDA
n_comp3 = min(3, n_classes - 1)
lda3 = LDA(n_components=n_comp3)
X_lda3 = lda3.fit_transform(X_scaled_df, y_enc)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_lda3[:,0], X_lda3[:,1], X_lda3[:,2] if n_comp3>2 else np.zeros_like(X_lda3[:,0]), c=y_enc, alpha=0.7)
ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_zlabel("LD3")
ax.set_title(f"LDA (3B) - Sınıf Ayrımı Odaklı İndirgeme")
plt.colorbar(sc, label="Air Quality (etiket)")
plt.show()

# 3B t-SNE
tsne3 = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=30, random_state=42)
X_tsne3 = tsne3.fit_transform(X_scaled_df)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_tsne3[:,0], X_tsne3[:,1], X_tsne3[:,2], c=y_enc, alpha=0.7)
ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2"); ax.set_zlabel("t-SNE-3")
ax.set_title("t-SNE (3B) - Doğrusal Olmayan İndirgeme")
plt.colorbar(sc, label="Air Quality (etiket)")
plt.show()

# =============================
# VERİ HAZIRLIĞI - TRAIN/TEST SPLIT + SMOTE
# =============================
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)


# y_test'i bir yere sabitle (ROC için yeni split yapacağız, overwrite olmayacak)
y_test_int = y_test.copy()

# SMOTE (sadece eğitim setine)
from imblearn.over_sampling import SMOTE
from collections import Counter
print("▶ SMOTE öncesi sınıf dağılımı:")
print(Counter(y_train))
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("\n▶ SMOTE sonrası sınıf dağılımı:")
print(Counter(y_train_balanced))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pd.Series(y_train).value_counts().plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title("SMOTE Öncesi"); axes[0].set_ylabel("Örnek Sayısı")
pd.Series(y_train_balanced).value_counts().plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title("SMOTE Sonrası"); axes[1].set_ylabel("Örnek Sayısı")
plt.tight_layout(); plt.show()

# CV stratejisi (dengelenmiş veri üzerinde kullanılacak)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =============================
# 1) LOJİSTİK REGRESYON
# =============================
from sklearn.linear_model import LogisticRegression
print("\n" + "="*60)
print("LOJİSTİK REGRESYON")
print("="*60)

lr_clf = LogisticRegression(max_iter=5000, random_state=42)
lr_params = {
    "clf__C": [0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "liblinear"]
}
pipe_lr = Pipeline([("clf", lr_clf)])
gs_lr = GridSearchCV(pipe_lr, lr_params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
# DİKKAT: Dengelenmiş eğitim verisi ile fit
gs_lr.fit(X_train_balanced, y_train_balanced)
print(f"En iyi parametreler: {gs_lr.best_params_}")
print(f"En iyi CV skoru: {gs_lr.best_score_:.4f}")

y_pred_lr = gs_lr.predict(X_test)
y_proba_lr = gs_lr.predict_proba(X_test)
print("\n▶ Test Performansı:")
print(f"Accuracy: {accuracy_score(y_test_int, y_pred_lr):.4f}")
print(f"Precision (macro): {precision_score(y_test_int, y_pred_lr, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test_int, y_pred_lr, average='macro'):.4f}")
print(f"F1 (macro): {f1_score(y_test_int, y_pred_lr, average='macro'):.4f}")
print(f"ROC AUC (OvR): {roc_auc_score(y_test_int, y_proba_lr, multi_class='ovr'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_int, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Lojistik Regresyon")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()

print("\n▶ Classification Report:")
print(classification_report(y_test_int, y_pred_lr))

# =============================
# 2) KARAR AĞACI
# =============================
from sklearn.tree import DecisionTreeClassifier
print("\n" + "="*60)
print("KARAR AĞACI")
print("="*60)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_params = {
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 5]
}
pipe_dt = Pipeline([("clf", dt_clf)])
gs_dt = GridSearchCV(pipe_dt, dt_params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
gs_dt.fit(X_train_balanced, y_train_balanced)
print(f"En iyi parametreler: {gs_dt.best_params_}")
print(f"En iyi CV skoru: {gs_dt.best_score_:.4f}")

y_pred_dt = gs_dt.predict(X_test)
print("\n▶ Test Performansı:")
print(f"Accuracy: {accuracy_score(y_test_int, y_pred_dt):.4f}")
print(f"Precision (macro): {precision_score(y_test_int, y_pred_dt, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test_int, y_pred_dt, average='macro'):.4f}")
print(f"F1 (macro): {f1_score(y_test_int, y_pred_dt, average='macro'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_int, y_pred_dt), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Karar Ağacı")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()

print("\n▶ Classification Report:")
print(classification_report(y_test_int, y_pred_dt))

# =============================
# 3) RASTGELE ORMAN
# =============================
from sklearn.ensemble import RandomForestClassifier
print("\n" + "="*60)
print("RASTGELE ORMAN")
print("="*60)

rf_clf = RandomForestClassifier(random_state=42)
rf_params = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2]
}
pipe_rf = Pipeline([("clf", rf_clf)])
gs_rf = GridSearchCV(pipe_rf, rf_params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
gs_rf.fit(X_train_balanced, y_train_balanced)
print(f"En iyi parametreler: {gs_rf.best_params_}")
print(f"En iyi CV skoru: {gs_rf.best_score_:.4f}")
y_pred_rf = gs_rf.predict(X_test)
y_proba_rf = gs_rf.predict_proba(X_test)
print("\n▶ Test Performansı:")
print(f"Accuracy: {accuracy_score(y_test_int, y_pred_rf):.4f}")
print(f"Precision (macro): {precision_score(y_test_int, y_pred_rf, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test_int, y_pred_rf, average='macro'):.4f}")
print(f"F1 (macro): {f1_score(y_test_int, y_pred_rf, average='macro'):.4f}")
print(f"ROC AUC (OvR): {roc_auc_score(y_test_int, y_proba_rf, multi_class='ovr'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_int, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Rastgele Orman")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()
print("\n▶ Classification Report:")
print(classification_report(y_test_int, y_pred_rf))

# Özellik önemleri
importances = gs_rf.best_estimator_.named_steps['clf'].feature_importances_
feature_names = X_scaled_df.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title("Özellik Önem Sıralaması (Rastgele Orman)")
plt.xlabel("Özellikler"); plt.ylabel("Önem Skoru")
plt.tight_layout(); plt.show()

print("\n▶ En önemli 5 özellik:")
for i in range(min(5, len(indices))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# =============================
# 4) SVM
# =============================
from sklearn.svm import SVC
print("\n" + "="*60)
print("DESTEK VEKTÖR MAKİNELERİ (SVM)")
print("="*60)

svm_clf = SVC(probability=True, random_state=42)
svm_params = {
    "clf__C": [0.5, 1, 5],
    "clf__kernel": ["rbf", "linear"],
    "clf__gamma": ["scale", "auto"]
}
pipe_svm = Pipeline([("clf", svm_clf)])
gs_svm = GridSearchCV(pipe_svm, svm_params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
gs_svm.fit(X_train_balanced, y_train_balanced)
print(f"En iyi parametreler: {gs_svm.best_params_}")
print(f"En iyi CV skoru: {gs_svm.best_score_:.4f}")

y_pred_svm = gs_svm.predict(X_test)
y_proba_svm = gs_svm.predict_proba(X_test)
print("\n▶ Test Performansı:")
print(f"Accuracy: {accuracy_score(y_test_int, y_pred_svm):.4f}")
print(f"Precision (macro): {precision_score(y_test_int, y_pred_svm, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test_int, y_pred_svm, average='macro'):.4f}")
print(f"F1 (macro): {f1_score(y_test_int, y_pred_svm, average='macro'):.4f}")
print(f"ROC AUC (OvR): {roc_auc_score(y_test_int, y_proba_svm, multi_class='ovr'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_int, y_pred_svm), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()
print("\n▶ Classification Report:")
print(classification_report(y_test_int, y_pred_svm))

# =============================
# 5) KNN
# =============================
from sklearn.neighbors import KNeighborsClassifier
print("\n" + "="*60)
print("K-EN YAKIN KOMŞULAR (KNN)")
print("="*60)

knn_clf = KNeighborsClassifier()
knn_params = {
    "clf__n_neighbors": [3, 5, 7, 11],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2]
}
pipe_knn = Pipeline([("clf", knn_clf)])
gs_knn = GridSearchCV(pipe_knn, knn_params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
gs_knn.fit(X_train_balanced, y_train_balanced)
print(f"En iyi parametreler: {gs_knn.best_params_}")
print(f"En iyi CV skoru: {gs_knn.best_score_:.4f}")

y_pred_knn = gs_knn.predict(X_test)
y_proba_knn = gs_knn.predict_proba(X_test)
print("\n▶ Test Performansı:")
print(f"Accuracy: {accuracy_score(y_test_int, y_pred_knn):.4f}")
print(f"Precision (macro): {precision_score(y_test_int, y_pred_knn, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test_int, y_pred_knn, average='macro'):.4f}")
print(f"F1 (macro): {f1_score(y_test_int, y_pred_knn, average='macro'):.4f}")
print(f"ROC AUC (OvR): {roc_auc_score(y_test_int, y_proba_knn, multi_class='ovr'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_int, y_pred_knn), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()
print("\n▶ Classification Report:")
print(classification_report(y_test_int, y_pred_knn))


# =============================
# 6) XGBoost
# =============================
from xgboost import XGBClassifier
print("\n" + "="*60)
print("XGBOOST")
print("="*60)

xgb_clf = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_params = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.1, 0.3],
    "clf__subsample": [0.8, 1.0]
}
pipe_xgb = Pipeline([("clf", xgb_clf)])
gs_xgb = GridSearchCV(pipe_xgb, xgb_params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
gs_xgb.fit(X_train_balanced, y_train_balanced)
print(f"En iyi parametreler: {gs_xgb.best_params_}")
print(f"En iyi CV skoru: {gs_xgb.best_score_:.4f}")

y_pred_xgb = gs_xgb.predict(X_test)
y_proba_xgb = gs_xgb.predict_proba(X_test)
print("\n▶ Test Performansı:")
print(f"Accuracy: {accuracy_score(y_test_int, y_pred_xgb):.4f}")
print(f"Precision (macro): {precision_score(y_test_int, y_pred_xgb, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test_int, y_pred_xgb, average='macro'):.4f}")
print(f"F1 (macro): {f1_score(y_test_int, y_pred_xgb, average='macro'):.4f}")
print(f"ROC AUC (OvR): {roc_auc_score(y_test_int, y_proba_xgb, multi_class='ovr'):.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_int, y_pred_xgb), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()
print("\n▶ Classification Report:")
print(classification_report(y_test_int, y_pred_xgb))

# =============================
# 7) MODELLERİN KARŞILAŞTIRMASI
# =============================
print("\n" + "="*60)
print("TÜM MODELLERİN KARŞILAŞTIRMASI")
print("="*60)
comparison = pd.DataFrame({
    "Model": ["Lojistik Regresyon", "Karar Ağacı", "Rastgele Orman", "SVM", "KNN", "XGBoost"],
    "CV Score": [gs_lr.best_score_, gs_dt.best_score_, gs_rf.best_score_,
                 gs_svm.best_score_, gs_knn.best_score_, gs_xgb.best_score_],
    "Accuracy": [accuracy_score(y_test_int, y_pred_lr), accuracy_score(y_test_int, y_pred_dt),
                 accuracy_score(y_test_int, y_pred_rf), accuracy_score(y_test_int, y_pred_svm),
                 accuracy_score(y_test_int, y_pred_knn), accuracy_score(y_test_int, y_pred_xgb)],
    "F1 (macro)": [f1_score(y_test_int, y_pred_lr, average='macro'),
                   f1_score(y_test_int, y_pred_dt, average='macro'),
                   f1_score(y_test_int, y_pred_rf, average='macro'),
                   f1_score(y_test_int, y_pred_svm, average='macro'),
                   f1_score(y_test_int, y_pred_knn, average='macro'),
                   f1_score(y_test_int, y_pred_xgb, average='macro')]
})
comparison = comparison.sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
print(comparison.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(comparison["Model"], comparison["CV Score"], color="steelblue")
axes[0].set_xlabel("5-Katlı CV Skoru"); axes[0].set_title("Cross-Validation Performansı"); axes[0].set_xlim(0, 1)
axes[1].barh(comparison["Model"], comparison["F1 (macro)"], color="coral")
axes[1].set_xlabel("F1 Score (macro)"); axes[1].set_title("Test Seti Performansı"); axes[1].set_xlim(0, 1)
plt.tight_layout(); plt.show()

print(f"\n✓ En iyi model: {comparison.iloc[0]['Model']}")
print(f"  F1 Score: {comparison.iloc[0]['F1 (macro)']:.4f}")

# =============================
# ROC EĞRİLERİ (AYRI DEĞİŞKEN İSİMLERİYLE, OVERWRITE YOK)
# =============================
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# One-hot
y_bin = label_binarize(y_enc, classes=np.unique(y_enc))
X_tr_roc, X_te_roc, y_tr_roc, y_te_roc = train_test_split(
    X_scaled_df, y_bin, test_size=0.2, random_state=42, stratify=y_enc
)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')
}

plt.figure(figsize=(10, 8))
for name, model in models.items():
    clf = OneVsRestClassifier(model)
    clf.fit(X_tr_roc, y_tr_roc)
    y_score = clf.predict_proba(X_te_roc)
    for i in range(y_te_roc.shape[1]):
        fpr, tpr, _ = roc_curve(y_te_roc[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f'{name} (class {i}, AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri (OvR - Multi-Class)')
plt.legend(loc="lower right"); plt.show()

# =============================
# PCA/LDA İndirgeme + RF Karşılaştırması
# =============================
from sklearn.metrics import accuracy_score, f1_score

# PCA -> RF
pca_reduced = PCA(n_components=5, random_state=42)
X_train_pca = pca_reduced.fit_transform(X_train_balanced)
X_test_pca = pca_reduced.transform(X_test)
rf_pca = RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42)
rf_pca.fit(X_train_pca, y_train_balanced)
y_pred_pca = rf_pca.predict(X_test_pca)
print(f"Orijinal veri - Accuracy: {accuracy_score(y_test_int, y_pred_rf):.4f}")
print(f"PCA (5 bileşen) - Accuracy: {accuracy_score(y_test_int, y_pred_pca):.4f}")
print(f"F1 Farkı: {f1_score(y_test_int, y_pred_rf, average='macro') - f1_score(y_test_int, y_pred_pca, average='macro'):.4f}")

# LDA -> RF (n_components <= sınıf sayısı - 1)
lda_nc = min(3, n_classes - 1)
lda_reduced = LDA(n_components=lda_nc)
X_train_lda = lda_reduced.fit_transform(X_train_balanced, y_train_balanced)
X_test_lda = lda_reduced.transform(X_test)
rf_lda = RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42)
rf_lda.fit(X_train_lda, y_train_balanced)
y_pred_lda = rf_lda.predict(X_test_lda)
print(f"LDA ({lda_nc} bileşen) - Accuracy: {accuracy_score(y_test_int, y_pred_lda):.4f}")

# =============================
# K-MEANS & DBSCAN
# =============================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_values = list(range(2, 11))
inertias = []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled_df)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_values, inertias, marker="o")
plt.xticks(k_values)
plt.title("Elbow (Dirsek) — Inertia vs k")
plt.xlabel("Küme sayısı (k)"); plt.ylabel("Inertia (Toplam İçi Varyans)")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

sil_scores = []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled_df)
    sil_scores.append(silhouette_score(X_scaled_df, labels))

plt.figure(figsize=(6,4))
plt.plot(k_values, sil_scores, marker="o")
plt.xticks(k_values)
plt.title("Silhouette Skoru vs k")
plt.xlabel("Küme sayısı (k)"); plt.ylabel("Ortalama Silhouette")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

k_opt = 4
kmeans = KMeans(n_clusters=k_opt, n_init=10, random_state=42)
cluster_labels_km = kmeans.fit_predict(X_scaled_df)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled_df)
plt.figure(figsize=(7,5))
sc = plt.scatter(X_pca2[:,0], X_pca2[:,1], c=cluster_labels_km, alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"K-Means (k={k_opt}) — PCA (2B)")
plt.colorbar(sc, label="Küme Etiketi"); plt.tight_layout(); plt.show()

try:
    contingency_km = pd.crosstab(cluster_labels_km, y_enc, rownames=["Cluster"], colnames=["True Label"])
    print(contingency_km)
except Exception as e:
    print("Gerçek etiket karşılaştırması atlandı:", e)

# DBSCAN
eps_val = 0.9
min_samples = 5
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled_df)
distances, indices = nbrs.kneighbors(X_scaled_df)
k_dists = np.sort(distances[:, -1])
plt.figure(figsize=(6,4))
plt.plot(k_dists)
plt.title(f"k-distance Grafiği (k={min_samples})")
plt.xlabel("Sıralı örnekler"); plt.ylabel(f"{min_samples}. komşuya uzaklık")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=eps_val, min_samples=min_samples)
cluster_labels_db = db.fit_predict(X_scaled_df)
unique, counts = np.unique(cluster_labels_db, return_counts=True)
print("Kümeler ve örnek sayıları:", dict(zip(unique, counts)))

# t-SNE (2B) + DBSCAN görselleştirme
tsne_db = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30, random_state=42)
X_tsne2 = tsne_db.fit_transform(X_scaled_df)
plt.figure(figsize=(7,5))
sc = plt.scatter(X_tsne2[:,0], X_tsne2[:,1], c=cluster_labels_db, alpha=0.7)
plt.title(f"DBSCAN — t-SNE (2B) | eps={eps_val}, min_samples={min_samples}")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.colorbar(sc, label="Küme Etiketi (-1: Noise)"); plt.tight_layout(); plt.show()

# PCA (2B) + DBSCAN görselleştirme
pca_db = PCA(n_components=2, random_state=42)
X_pca_db = pca_db.fit_transform(X_scaled_df)
plt.figure(figsize=(7,5))
sc = plt.scatter(X_pca_db[:,0], X_pca_db[:,1], c=cluster_labels_db, alpha=0.7)
plt.title(f"DBSCAN — PCA (2B) | eps={eps_val}, min_samples={min_samples}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(sc, label="Küme Etiketi (-1: Noise)"); plt.tight_layout(); plt.show()

# Farklı eps değerleri
results = []
for eps in [0.5, 0.7, 0.9, 1.1, 1.3]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    results.append({'eps': eps, 'clusters': n_clusters, 'noise_ratio': n_noise / len(labels)})
results_df = pd.DataFrame(results)
print(results_df)

try:
    contingency_db = pd.crosstab(cluster_labels_db, y_enc, rownames=["DBSCAN Cluster"], colnames=["True Label"])
    print(contingency_db)
except Exception as e:
    print("Gerçek etiket karşılaştırması atlandı:", e)
