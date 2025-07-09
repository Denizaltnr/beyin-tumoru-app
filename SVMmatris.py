import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Verilen classification report değerleri (manuel girildi)
data = {
    'precision': [0.98, 0.89, 0.97, 0.96],
    'recall':    [0.94, 0.93, 0.90, 0.99],
    'f1-score':  [0.96, 0.91, 0.93, 0.98],
    'support':   [165, 165, 79, 165]
}

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

df = pd.DataFrame(data, index=labels)

# Support değerleri çok farklı aralıklarda olduğu için görselde renklendirmeden çıkarabiliriz ya da normalize edebiliriz.
# Burada sadece precision, recall, f1-score görselleştiriliyor.

plt.figure(figsize=(10,6))
sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap='Blues', fmt=".2f")

plt.title('SVM Modeli - Sınıflandırma Performans Metrikleri')
plt.ylabel('Sınıflar')
plt.show()
