# Kütüphaneler
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# 1. Iris veri setini yüklüyoruz
iris = load_iris()


# 2. Verileri DataFrame'e çeviriyoruz
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Tahmin etmek istediğimiz çiçek türünü ekliyoruz
df["target"] = iris.target


# 3. Veriyi inceliyoruz
print(df.head())
print("\nVeri boyutu:", df.shape)
print("\nSiniflar:", iris.target_names)


# 4. X ve y'yi ayırıyoruz

# X = Modelin bakacağı özellikler
X = df.drop("target", axis=1)

# y = Modelin tahmin etmeye çalışacağı cevap
y = df["target"]


# 5. Train ve Test olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(X_train_scaled, y_train)
logistic_tahmin = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(y_test, logistic_tahmin)


# Random Forest
forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)
train_tahmin = forest_model.predict(X_train)
test_tahmin = forest_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_tahmin)
test_accuracy = accuracy_score(y_test, test_tahmin)

print("\n--- Overfitting Kontrolu ---")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

forest_tahmin = forest_model.predict(X_test)

forest_accuracy = accuracy_score(y_test, forest_tahmin)


print("\nLogistic Regression Accuracy:", logistic_accuracy)
print("Random Forest Accuracy:", forest_accuracy)
cv_skorlari = cross_val_score(
    forest_model,
    X,
    y,
    cv=5
)

print("\n5-Fold Cross Validation Sonuclari:")
print(cv_skorlari)
print("Ortalama Accuracy:", cv_skorlari.mean())

# Kontrol edelim
print("\nTrain veri sayisi:", len(X_train))
print("Test veri sayisi:", len(X_test))
print("\n--- Yeni Cicek Tahmini ---")

sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

yeni_cicek = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)

tahmin = forest_model.predict(yeni_cicek)

print("\nTahmin edilen cicek:", iris.target_names[tahmin[0]])

