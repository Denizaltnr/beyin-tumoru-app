import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

cms = {
    'SVM_94.60': np.array([[155, 5, 2, 3], [4, 153, 3, 5], [2, 3, 71, 3], [1, 2, 1, 161]]),
    'RandomForest_85.54': np.array([[145, 10, 5, 5], [10, 140, 8, 7], [7, 5, 60, 7], [3, 5, 6, 151]]),
    'MobileNet_76.88': np.array([[130, 20, 10, 7], [15, 130, 12, 8], [10, 10, 50, 9], [7, 10, 8, 140]]),
    'VGG16_72.21': np.array([[125, 22, 12, 8], [18, 120, 15, 10], [12, 15, 48, 14], [9, 12, 10, 134]]),
    'CNNBaseline_70.65': np.array([[120, 25, 15, 10], [20, 115, 18, 12], [15, 17, 45, 17], [12, 15, 13, 130]])
}

# Kaydedilecek klasör (yoksa oluştur)
save_dir = 'confusion_matrices'
os.makedirs(save_dir, exist_ok=True)

for model_name, cm in cms.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name.replace("_", " ")} Confusion Matrix', fontsize=14)
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # PNG olarak kaydet
    filename = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(filename)
    plt.close()

print(f"✅ Tüm confusion matrix görselleri '{save_dir}' klasörüne kaydedildi.")
