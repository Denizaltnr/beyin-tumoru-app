import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Veri yolu ve sınıflar
data_dir = 'C:/Users/ben_d/Desktop/VeriProje/Brain-Tumor-Classification-MRI/Training'
class_names = sorted(os.listdir(data_dir))
print("Sınıflar:", class_names)

# 2. MobileNet model (özellik çıkarımı için)
img_size = (224, 224)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 3. Özellik çıkarma fonksiyonu
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

# 4. Tüm verinin özelliklerini çıkar
X, y = [], []
for label in class_names:
    label_folder = os.path.join(data_dir, label)
    for file in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file)
        try:
            feat = extract_features(file_path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Hata oluştu: {file_path} => {e}")

X = np.array(X)
y = np.array(y)

print("Özellik matrisi boyutu:", X.shape)
print("Etiket sayısı:", len(y))

# 5. PCA ile 2 bileşene indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 6. t-SNE ile 2 bileşene indirgeme
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=300)
X_tsne = tsne.fit_transform(X)

# 7. Görselleştirme fonksiyonu
def plot_embedding(X_embedded, title):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(y)
    for label in unique_labels:
        idx = np.where(y == label)
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

# 8. PCA görselleştirme
plot_embedding(X_pca, "PCA ile Özellik Uzayı Görselleştirme")

# 9. t-SNE görselleştirme
plot_embedding(X_tsne, "t-SNE ile Özellik Uzayı Görselleştirme")
