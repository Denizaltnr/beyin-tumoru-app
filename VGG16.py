import os
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling, Flatten, Dense, Dropout
from keras.models import Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 📁 Klasör yolları
base_dir = 'C:/Users/ben_d/Desktop/VeriProje/Brain-Tumor-Classification-MRI'
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Testing')

img_size = (224, 224)
batch_size = 32

# 🔤 Sınıfları tanımla
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f'Sınıflar: {class_names}')

# 📦 Eğitim ve doğrulama veri setleri
train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# 📦 Test veri seti
test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size
)

# 🔄 Normalize et
normalization_layer = Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# 🧠 VGG16 Modeli
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
for layer in base_model.layers:
    layer.trainable = False  # Tüm katmanları dondur

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ⚙️ Modeli derle
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 🛑 Erken durdurma
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 🚀 Modeli eğit
epochs = 25
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stop]
)

# 💾 Modeli kaydet
model.save("vgg16_brain_tumor_model.h5")
print("✅ Model kaydedildi: vgg16_brain_tumor_model.h5")

# 📊 Eğitim / Doğrulama grafikleri
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()

# 🧪 Test verisinde değerlendirme
test_loss, test_acc = model.evaluate(test_dataset)
print("📊 Test Doğruluğu: {:.2f}%".format(test_acc * 100))
