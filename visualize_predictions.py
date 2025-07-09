import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# 🔹 1. Veri yolu ve sınıflar
data_dir = 'C:/Users/ben_d/Desktop/VeriProje/Brain-Tumor-Classification-MRI/Training'
class_names = sorted(os.listdir(data_dir))
img_size = (224, 224)

# 🔹 2. Özellik çıkarımı için MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

# 🔹 3. Tüm veri yolları ve etiketleri
image_paths = []
labels = []

for label in class_names:
    label_folder = os.path.join(data_dir, label)
    for filename in os.listdir(label_folder):
        image_paths.append(os.path.join(label_folder, filename))
        labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

# 🔹 4. Veriyi ayır
_, test_paths, _, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 🔹 5. Eğitilmiş modeli yükle (SVM)
svm_model = joblib.load('svm_model.pkl')

# 🔹 6. Örnek görseller üzerinden tahmin yap ve görselleştir
num_samples = 8  # Kaç örnek gösterilsin?
plt.figure(figsize=(15, 8))

for i in range(num_samples):
    img_path = test_paths[i]
    true_label = test_labels[i]
    features = extract_features(img_path)
    prediction = svm_model.predict([features])[0]

    # Görseli oku ve çiz
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Gerçek: {true_label}\nTahmin: {prediction}", color='green' if true_label == prediction else 'red')

plt.suptitle("📷 SVM Modeli ile Test Görselleri Tahmini", fontsize=16)
plt.tight_layout()
plt.show()
