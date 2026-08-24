import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)

veriler = {
    "calisma_saati": [1, 2, 3, 4, 5, 6, 1.5, 2.5, 3.5, 4.5,
                      5.5, 6.5, 2, 3, 4, 5, 6, 7, 1, 7.5],

    "uyku_saati": [5, 6, 6, 7, 7, 8, 5, 6, 7, 7,
                   8, 8, 5, 6, 7, 8, 6, 8, 4, 7],

    "devamsizlik": [12, 9, 7, 5, 3, 2, 11, 8, 6, 4,
                    3, 1, 10, 6, 5, 2, 4, 0, 14, 1],

    "gecti_mi": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(veriler)

X = df[["calisma_saati", "uyku_saati", "devamsizlik"]]
y = df["gecti_mi"]

print(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

tahminler = model.predict(X_test)

print("Gerçek sonuçlar:", y_test.values)
print("Model tahminleri:", tahminler)

accuracy = accuracy_score(y_test, tahminler)
cm = confusion_matrix(y_test, tahminler)
precision = precision_score(y_test, tahminler)
recall = recall_score(y_test, tahminler)

print("Accuracy:", accuracy)

print("Confusion Matrix:")
print(cm)

print("Precision:", precision)
print("Recall:", recall)
plt.figure(figsize=(12, 7))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Kaldi", "Gecti"],
    filled=True
)

plt.show()