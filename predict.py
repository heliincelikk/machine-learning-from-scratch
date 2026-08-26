import joblib
import pandas as pd

model = joblib.load("svm_model.pkl")

yeni_musteri = pd.DataFrame([{
    "yas": 33,
    "gelir": 70000,
    "deneyim_yili": 6,
    "sehir_Ankara": 0,
    "sehir_Antalya": 1,
    "sehir_Istanbul": 0
}])

tahmin = model.predict(yeni_musteri)

print("Tahmin:", tahmin[0])

if tahmin[0] == 1:
    print("Musteri urunu alabilir.")
else:
    print("Musteri urunu almayabilir.")