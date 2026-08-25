import pandas as pd

veriler = {
    "yas": [25, 31, None, 42, 36],
    "gelir": [45000, None, 65000, 90000, 72000],
    "sehir": ["Istanbul", "Antalya", "Istanbul", None, "Ankara"],
    "kredi_onay": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(veriler)

print(df)

print("\nEksik veri sayilari:")
print(df.isnull().sum())
print(df)

print("\nEksik veri sayilari:")
print(df.isnull().sum())

# Eksik sayisal verileri doldur
df["yas"] = df["yas"].fillna(df["yas"].median())
df["gelir"] = df["gelir"].fillna(df["gelir"].median())

# Eksik kategorik veriyi doldur
df["sehir"] = df["sehir"].fillna(df["sehir"].mode()[0])

# Sonucu kontrol et
print("\nEksik veriler dolduruldu:")
print(df)

print("\nKalan eksik veri:")
print(df.isnull().sum())
df = pd.get_dummies(
    df,
    columns=["sehir"],
    dtype=int
)

print("\nEncoding sonrasi:")
print(df)