import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri klasörleri
data_dir = r"C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

# Sınıflar ve label map (sabit sıra için sorted kullandık)
categories = sorted(os.listdir(train_dir))
label_map = {cat: idx for idx, cat in enumerate(categories)}
num_classes = len(categories)

print("Sınıflar:", categories)

def load_data(folder, img_size):
    data = []
    labels = []
    for cat in categories:
        path = os.path.join(folder, cat)
        label = label_map[cat]
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"UYARI: Bozuk görüntü atlandı: {img_path}")
                continue
            # Modelin istediği input shape e göre resize
            img = cv2.resize(img, (img_size[1], img_size[0]))  # cv2.resize (width, height) alır
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

# Model yolları ve isimleri
model_paths = {
    "CNN": "brain_tumor_cnn_model.h5",
    "MobileNet": "mobilenet_brain_tumor_model.h5",
    "RandomForest": "random_forest_model.h5",  # Bunlar sklearn modelleri, ayrı yüklenmeli
    "SVM": "svm_model.h5",
    "VGG16": "vgg16_brain_tumor_model.h5"
}

results = {}

for name, path in model_paths.items():
    print(f"\nModel {name} yükleniyor...")
    
    # Sklearn modelleri için yükleme farklıdır, burada sadece keras modelleri ele alıyoruz
    if name in ["RandomForest", "SVM"]:
        print(f"{name} sklearn modeli, burada keras ile yüklenemez. Farklı yöntemle yüklemelisin.")
        continue
    
    try:
        model = load_model(path)
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        continue
    
    # Modelin beklediği input shape'i al (ör: (None, 224, 224, 3))
    input_shape = model.input_shape
    print(f"{name} model input shape: {input_shape}")
    
    # input_shape: (None, height, width, channels)
    _, height, width, channels = input_shape
    
    # Test verisini modelin beklediği input shape ile yükle
    X_test, y_test = load_data(test_dir, (height, width))
    X_test = X_test / 255.0  # normalize et
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"Test verisi shape: {X_test.shape}")
    
    try:
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
    except Exception as e:
        print(f"{name} modelinde tahmin yaparken hata oluştu: {e}")
        continue
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Doğruluğu: {acc*100:.2f}%")
    
    print(f"{name} Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:\n{cm}")
    
    results[name] = acc

# Performans karşılaştırma grafiği
plt.figure(figsize=(8,5))
plt.bar(results.keys(), [v*100 for v in results.values()], color='skyblue')
plt.title("Modellerin Test Doğrulukları (%)")
plt.ylabel("Doğruluk (%)")
plt.ylim(0, 100)
plt.show()
