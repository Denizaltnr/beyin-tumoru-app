import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 🔹 Klasör yolları
data_dir = r"C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

# 🔹 Sınıfları al
categories = os.listdir(train_dir)
num_classes = len(categories)
print("Sınıflar:", categories)

# 🔹 Görselleri yükle
IMG_SIZE = 150

def load_data(folder):
    data, labels = [], []
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    
    for cat in categories:
        path = os.path.join(folder, cat)
        label = label_map[cat]
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(label)
            except Exception as e:
                print("Hatalı görüntü:", img_path, e)
                
    return np.array(data), np.array(labels)

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# 🔹 Normalize et
X_train, X_test = X_train / 255.0, X_test / 255.0

# 🔹 Kategorik hale getir
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 🔹 Eğitim / doğrulama ayır
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_cat, test_size=0.2, random_state=42)

# 🔹 CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🔹 Modeli eğit
history = model.fit(
    X_train_split, y_train_split,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# 🔹 Eğitim & Doğrulama Doğruluğu Grafiği
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', marker='o')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', marker='o')
plt.title('Doğruluk (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# 🔹 Eğitim & Doğrulama Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı', marker='o')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', marker='o')
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 🔹 Test üzerinde değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print("📊 Test Doğruluğu: {:.2f}%".format(test_acc * 100))

# 🔹 Modeli Kaydet
model.save("brain_tumor_cnn_model.h5")
print("✅ Model 'brain_tumor_cnn_model.h5' olarak kaydedildi.")
