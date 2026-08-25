import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("musteriler.csv")

print(df)

print("\nVeri boyutu:")
print(df.shape)

print("\nEksik veriler:")
print(df.isnull().sum())
df = pd.get_dummies(
    df,
    columns=["sehir"],
    dtype=int
)

print("\nEncoding sonrasi:")
print(df)
X = df.drop("urun_aldi", axis=1)
y = df["urun_aldi"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain veri sayisi:", len(X_train))
print("Test veri sayisi:", len(X_test))