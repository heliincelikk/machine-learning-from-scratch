import pandas as pd
from sklearn.linear_model import LinearRegression

veriler = {
    "calisma_saati": [1, 2, 3, 4, 5],
    "sinav_puani": [40, 50, 60, 70, 80]
}

df = pd.DataFrame(veriler)

print(df)

X = df[["calisma_saati"]]
y = df["sinav_puani"]

model = LinearRegression()

model.fit(X, y)

yeni_veri = pd.DataFrame(
    {"calisma_saati": [6]}
)

tahmin = model.predict(yeni_veri)

print("Tahmin edilen sınav puanı:", tahmin[0])
print("Katsayı:", model.coef_[0])
print("Sabit:", model.intercept_)