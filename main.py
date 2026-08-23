import pandas as pd

veriler = {
    "calisma_saati": [1, 2, 3, 4, 5],
    "sinav_puani": [40, 50, 60, 70, 80]
}

df = pd.DataFrame(veriler)

print(df)