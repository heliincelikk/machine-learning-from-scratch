# 1. KÜTÜPHANELER
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 2. VERİLERİMİZ
veriler = {
    "calisma_saati": [1, 2, 3, 4, 5, 6, 1.5, 2.5, 3.5, 4.5,
                      5.5, 6.5, 2, 3, 4, 5, 6, 7, 1, 7.5],

    "uyku_saati": [5, 6, 6, 7, 7, 8, 5, 6, 7, 7,
                   8, 8, 5, 6, 7, 8, 6, 8, 4, 7],

    "devamsizlik": [12, 9, 7, 5, 3, 2, 11, 8, 6, 4,
                    3, 1, 10, 6, 5, 2, 4, 0, 14, 1],

    "sinav_puani": [35, 45, 55, 66, 76, 86, 40, 50, 61, 71,
                    82, 91, 43, 57, 68, 79, 84, 95, 31, 96]
}


# 3. VERİYİ TABLOYA ÇEVİRİYORUZ
df = pd.DataFrame(veriler)

print(df)


# 4. X VE y'Yİ AYIRIYORUZ
X = df[["calisma_saati", "uyku_saati", "devamsizlik"]]
y = df["sinav_puani"]

# 5. TRAIN VE TEST OLARAK AYIRIYORUZ
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)

tahminler = model.predict(X_test)

print("Gerçek puanlar:")
print(y_test.values)

print("Modelin tahminleri:")
print(tahminler)
mae = mean_absolute_error(y_test, tahminler)
r2 = r2_score(y_test, tahminler)
print("Katsayılar:", model.coef_)
print("Sabit:", model.intercept_)

print("Ortalama hata (MAE):", mae)
print("R² skoru:", r2)
plt.scatter(X_test["calisma_saati"], y_test, label="Gercek Puan")

plt.scatter(
    X_test["calisma_saati"],
    tahminler,
    label="Model Tahmini"
)

plt.xlabel("Calisma Saati")
plt.ylabel("Sinav Puani")
plt.title("Gercek Degerler vs Model Tahminleri")

plt.legend()
# plt.show()