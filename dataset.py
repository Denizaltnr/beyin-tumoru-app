import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Sonuçlar (senin verdiğin)
accuracy = 0.7974
precision = 0.8292
recall = 0.7974
f1 = 0.7714

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

classification_report_dict = {
    'glioma_tumor': {'precision': 0.94, 'recall': 0.33, 'f1-score': 0.49, 'support': 91},
    'meningioma_tumor': {'precision': 0.73, 'recall': 0.94, 'f1-score': 0.82, 'support': 115},
    'no_tumor': {'precision': 0.75, 'recall': 0.96, 'f1-score': 0.85, 'support': 105},
    'pituitary_tumor': {'precision': 0.96, 'recall': 0.92, 'f1-score': 0.94, 'support': 74}
}

conf_matrix = np.array([
    [30, 33, 25, 3],
    [2, 108, 5, 0],
    [0, 4, 101, 0],
    [0, 3, 3, 68]
])

# Kayıt klasörü
save_dir = r"C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI\Dataset"
os.makedirs(save_dir, exist_ok=True)

# 1. Genel metriklerin bar grafiği
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8,5))
sns.barplot(x=metrics, y=values, palette='viridis')
plt.ylim(0,1)
plt.title('Genel Model Değerlendirme Metrikleri')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.savefig(os.path.join(save_dir, "genel_metrikler.png"))
plt.close()

# 2. Sınıf bazlı precision, recall, f1-score bar grafiği
metrics_per_class = ['precision', 'recall', 'f1-score']
plt.figure(figsize=(12,7))

for i, metric in enumerate(metrics_per_class):
    plt.subplot(1, 3, i+1)
    scores = [classification_report_dict[cls][metric] for cls in class_names]
    sns.barplot(x=class_names, y=scores, palette='magma')
    plt.ylim(0,1)
    plt.title(f'Sınıf Bazında {metric.capitalize()}')
    plt.xticks(rotation=30)
    for j, v in enumerate(scores):
        plt.text(j, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sinif_bazli_metrikler.png"))
plt.close()

# 3. Confusion Matrix ısı haritası
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

print(f"Görselleştirmeler '{save_dir}' klasörüne kaydedildi.")
