import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

veriler = {
    "calisma_saati": [1, 2, 1.5, 2.5, 3, 3.5, 5, 5.5, 6, 6.5, 7, 7.5],
    "devamsizlik": [14, 12, 13, 10, 9, 8, 4, 3, 2, 2, 1, 0]
}

df = pd.DataFrame(veriler)

X = df[["calisma_saati", "devamsizlik"]]

model = KMeans(
    n_clusters=2,
    random_state=42
)

model.fit(X)

df["grup"] = model.labels_

print(df)
plt.scatter(
    df["calisma_saati"],
    df["devamsizlik"],
    c=df["grup"]
)

plt.xlabel("Calisma Saati")
plt.ylabel("Devamsizlik")
plt.title("K-Means Ogrenci Gruplari")

plt.show()
