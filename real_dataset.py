import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib


# 1. CSV dosyasını oku
df = pd.read_csv("musteriler.csv")

print(df)

print("\nVeri boyutu:")
print(df.shape)

print("\nEksik veriler:")
print(df.isnull().sum())


# 2. Kategorik veriyi sayısal hale getir
df = pd.get_dummies(
    df,
    columns=["sehir"],
    dtype=int
)

print("\nEncoding sonrasi:")
print(df)


# 3. X ve y'yi ayır
X = df.drop("urun_aldi", axis=1)
y = df["urun_aldi"]


# 4. Train / Test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain veri sayisi:", len(X_train))
print("Test veri sayisi:", len(X_test))


# -------------------------------------------------
# 5. LOGISTIC REGRESSION
# -------------------------------------------------

logistic_model = LogisticRegression(max_iter=1000)

# Model öğreniyor
logistic_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin
logistic_tahminler = logistic_model.predict(X_test)

# Başarı
logistic_accuracy = accuracy_score(
    y_test,
    logistic_tahminler
)

print("\n--- Logistic Regression ---")
print("Gercek sonuclar:", y_test.values)
print("Tahminler:", logistic_tahminler)
print("Accuracy:", logistic_accuracy)


# -------------------------------------------------
# 6. SCALING
# -------------------------------------------------

scaler = StandardScaler()

# Train verisinden ölçeklendirmeyi öğren
# ve train verisini dönüştür
X_train_scaled = scaler.fit_transform(X_train)

# Test verisine sadece aynı dönüşümü uygula
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# 7. KNN
# -------------------------------------------------

knn_model = KNeighborsClassifier(
    n_neighbors=3
)

# KNN öğreniyor
knn_model.fit(
    X_train_scaled,
    y_train
)

# Tahmin
knn_tahminler = knn_model.predict(
    X_test_scaled
)

# Başarı
knn_accuracy = accuracy_score(
    y_test,
    knn_tahminler
)

print("\n--- KNN ---")
print("Gercek sonuclar:", y_test.values)
print("Tahminler:", knn_tahminler)
print("Accuracy:", knn_accuracy)

# -------------------------------------------------
# 9. GRID SEARCH - KNN
# -------------------------------------------------

parametreler = {
    "n_neighbors": [3, 5, 7, 9]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    parametreler,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train_scaled, y_train)

print("\n--- GridSearchCV ---")
print("En iyi K:", grid_search.best_params_)
print("En iyi CV Accuracy:", grid_search.best_score_)
# -------------------------------------------------
# 8. SVM
# -------------------------------------------------

svm_model = SVC()

# Scaling yapılmış train verisiyle öğren
svm_model.fit(X_train_scaled, y_train)

# Test verisini tahmin et
svm_tahminler = svm_model.predict(X_test_scaled)

# Başarıyı ölç
svm_accuracy = accuracy_score(y_test, svm_tahminler)

print("\n--- SVM ---")
print("Gercek sonuclar:", y_test.values)
print("Tahminler:", svm_tahminler)
print("Accuracy:", svm_accuracy)
# -------------------------------------------------
# 10. GRID SEARCH - SVM
# -------------------------------------------------

svm_parametreler = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

svm_grid = GridSearchCV(
    SVC(),
    svm_parametreler,
    cv=5,
    scoring="accuracy"
)

svm_grid.fit(X_train_scaled, y_train)

print("\n--- SVM GridSearchCV ---")
print("En iyi parametreler:", svm_grid.best_params_)
print("En iyi CV Accuracy:", svm_grid.best_score_)
# -------------------------------------------------
# 11. PIPELINE
# -------------------------------------------------

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# Pipeline'i eğit
svm_pipeline.fit(X_train, y_train)
joblib.dump(svm_pipeline, "svm_model.pkl")

print("\nModel kaydedildi: svm_model.pkl")

# Tahmin yap
pipeline_tahminler = svm_pipeline.predict(X_test)

# Başarı
pipeline_accuracy = accuracy_score(
    y_test,
    pipeline_tahminler
)

print("\n--- SVM Pipeline ---")
print("Gercek sonuclar:", y_test.values)
print("Tahminler:", pipeline_tahminler)
print("Accuracy:", pipeline_accuracy)