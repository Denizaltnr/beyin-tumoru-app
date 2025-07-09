import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🔹 1. Veri yolu ve sınıflar
data_dir = 'C:/Users/ben_d/Desktop/VeriProje/Brain-Tumor-Classification-MRI/Training'
class_names = sorted(os.listdir(data_dir))
print("Sınıflar:", class_names)

# 🔹 2. Model tanımı (MobileNet, sadece özellik çıkarımı için)
img_size = (224, 224)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 🔹 3. Özellik çıkarım fonksiyonu
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

# 🔹 4. Özellikleri oluştur
X, y = [], []

for label in class_names:
    label_folder = os.path.join(data_dir, label)
    for file in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file)
        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Hata oluştu: {file_path} => {e}")

X = np.array(X)
y = np.array(y)
print("Özellik boyutu:", X.shape)

# 🔹 5. Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 6A. SVM Modeli
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)

print("\n📊 SVM Sonuçları")
print("Doğruluk: {:.2f}%".format(svm_acc * 100))
print(classification_report(y_test, svm_preds))

# 🔹 6B. Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print("\n📊 Random Forest Sonuçları")
print("Doğruluk: {:.2f}%".format(rf_acc * 100))
print(classification_report(y_test, rf_preds))

# 🔹 7. Grafik: Karşılaştırma
models = ['SVM', 'Random Forest']
accuracies = [svm_acc * 100, rf_acc * 100]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies, color=['steelblue', 'orange'])
plt.ylim(0, 100)
plt.ylabel('Test Doğruluk (%)')
plt.title('SVM vs Random Forest Doğruluk Karşılaştırması')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height - 5, f'{height:.2f}%', ha='center', color='white', fontsize=12)

plt.tight_layout()
plt.show()

# 🔹 8. Modelleri Kaydet
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')

print("\n✅ Modeller kaydedildi: svm_model.pkl & random_forest_model.pkl")
