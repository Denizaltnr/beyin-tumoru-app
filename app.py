try:
    import tensorflow as tf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.12.0"])
    import tensorflow as tf



import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import os  # 🔹 BU SATIRI EKLE
import glob
from tensorflow.keras.models import load_model
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Sayfa ayarları
st.set_page_config(page_title="Beyin Tümörü Sınıflandırma", layout="wide")
st.title("🧠 Beyin Tümörü Sınıflandırma ve Model Karşılaştırma Arayüzü")

# Model dosya yolları
svm_model_path = "svm_model.pkl"
rf_model_path = "random_forest_model.pkl"
mobilenet_model_path = "mobilenet_brain_tumor_model.h5"
vgg16_model_path = "vgg16_brain_tumor_model.h5"
cnn_model_path = "brain_tumor_cnn_model.h5"

# MODELLERİ YÜKLE
if "svm_model" not in st.session_state:
    st.session_state.svm_model = joblib.load(svm_model_path)
    st.session_state.rf_model = joblib.load(rf_model_path)
    st.session_state.mobilenet_model = load_model(mobilenet_model_path)
    st.session_state.vgg16_model = load_model(vgg16_model_path)
    st.session_state.cnn_model = load_model(cnn_model_path)

# MobileNet'i sadece özellik çıkarımı için yükle
if "mobilenet_feature_extractor" not in st.session_state:
    st.session_state.mobilenet_feature_extractor = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Menü seçenekleri
menu = st.sidebar.selectbox("Menü Seç", [
    "🏁 Giriş",
    "📊 Model Karşılaştırması",
    "📷 Görsel Yükle ve Tahmin Al",
    "📈 PCA / t-SNE Görselleştirme",
    "📋 Rapor & Matrisler"
])

# Sınıf etiketleri
labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# 1. GİRİŞ
if menu == "🏁 Giriş":
    st.header("Proje Hakkında")
    st.write("""
        Bu uygulama, beyin tümörü sınıflandırması için eğitilmiş modelleri karşılaştırmalı olarak sunar ve kullanıcıların kendi MRI görüntüleri üzerinde test yapmasına olanak tanır.
        
        **Kullanılan Modeller:**
        - CNN
        - MobileNet (Öznitelik çıkarımı)
        - VGG16
        - SVM
        - Random Forest
        
        Ayrıca PCA veya t-SNE ile öznitelik uzayı görselleştirmeleri, sınıflandırma raporları ve örnek tahminler de sağlanır.
    """)

# 2. MODEL KARŞILAŞTIRMA
elif menu == "📊 Model Karşılaştırması":
    st.subheader("Modellerin Doğruluk Oranları")
    model_scores = {
        "CNN": 70.65,
        "MobileNet": 76.88,
        "VGG16": 72.21,
        "SVM": 94.60,
        "Random Forest": 85.54
    }
    df_scores = pd.DataFrame(list(model_scores.items()), columns=["Model", "Doğruluk (%)"])
    best_model = df_scores.loc[df_scores["Doğruluk (%)"].idxmax(), "Model"]
    colors = ['lightgreen' if model == best_model else 'skyblue' for model in df_scores["Model"]]

    fig, ax = plt.subplots()
    bars = ax.bar(df_scores["Model"], df_scores["Doğruluk (%)"], color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title("Model Doğruluk Karşılaştırması")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.2f}%", ha='center', fontsize=9)

    st.pyplot(fig)
    st.success(f"✅ En yüksek doğruluk: **{best_model}** ({model_scores[best_model]}%)")

# 3. GÖRSEL YÜKLE VE TAHMİN AL
elif menu == "📷 Görsel Yükle ve Tahmin Al":
    st.subheader("MRI Görseli ile Tahmin")

    model_options = {
        "CNN": "cnn_model",
        "MobileNet": "mobilenet_model",
        "VGG16": "vgg16_model",
        "SVM": "svm_model",
        "Random Forest": "rf_model"
    }
    selected_model_name = st.selectbox("Model Seçin", list(model_options.keys()))
    selected_model_key = model_options[selected_model_name]

    uploaded_file = st.file_uploader("MRI Görseli Yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Yüklenen Görsel", use_column_width=True)

        model = st.session_state[selected_model_key]

        # 🔸 SVM veya RF ise MobileNet ile özellik çıkar
        if selected_model_name in ["SVM", "Random Forest"]:
            img_resized = cv2.resize(image, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = img_to_array(img_rgb)
            img_expanded = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_expanded)

            feature_extractor = st.session_state.mobilenet_feature_extractor
            features = feature_extractor.predict(img_preprocessed, verbose=0).flatten().reshape(1, -1)

            pred_label = model.predict(features)[0]  # string döner
            st.write(f"**{selected_model_name} Model Tahmini:** {pred_label}")

        else:
            if selected_model_name == "CNN":
                img_resized = cv2.resize(image, (150, 150))
            else:
                img_resized = cv2.resize(image, (224, 224))

            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb / 255.0
            features = np.expand_dims(img_norm, axis=0)

            preds = model.predict(features)
            pred_idx = np.argmax(preds, axis=1)[0]
            pred_label = labels[pred_idx]

            st.write(f"**{selected_model_name} Model Tahmini:** {pred_label}")


# 4. PCA / t-SNE GÖRSELLEŞTİRME
elif menu == "📈 PCA / t-SNE Görselleştirme":
    st.subheader("PCA / t-SNE ile Öznitelik Uzayı Görselleştirme")
    st.info("MobileNet ile çıkarılan özniteliklerin 2 boyutlu PCA veya t-SNE dönüşümü bu kısımda başlatılabilir.")

    uploaded_folder = st.text_input("🔍 Özellik çıkarmak için eğitim verisinin klasör yolunu girin:", "dataset/test")

    if st.button("🧠 Özellikleri MobileNet ile Çıkar ve Görselleştir"):
        import glob
        from tqdm import tqdm

        feature_extractor = st.session_state.mobilenet_feature_extractor

        X = []
        y = []

        for label in labels:
            image_paths = glob.glob(os.path.join(uploaded_folder, label, "*.jpg"))
            for img_path in tqdm(image_paths, desc=f"{label}"):
                image = cv2.imread(img_path)
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)

                features = feature_extractor.predict(image, verbose=0)
                X.append(features.flatten())
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
        X_tsne = tsne.fit_transform(X)

        # Kaydet
        np.save("mobilenet_features.npy", X)
        np.save("mobilenet_labels.npy", y)

        st.success("✅ Özellik çıkarımı tamamlandı ve PCA / t-SNE uygulanıyor.")

        def plot_embedding(X_embedded, title):
            fig, ax = plt.subplots()
            unique_labels = np.unique(y)
            colors = plt.cm.get_cmap("tab10", len(unique_labels))

            for i, label in enumerate(unique_labels):
                indices = np.where(y == label)
                ax.scatter(X_embedded[indices, 0], X_embedded[indices, 1],
                           label=label, alpha=0.7, color=colors(i))
            ax.set_title(title)
            ax.set_xlabel("Bileşen 1")
            ax.set_ylabel("Bileşen 2")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        plot_embedding(X_pca, "📊 PCA ile Özellik Uzayı Görselleştirmesi")
        plot_embedding(X_tsne, "🌐 t-SNE ile Özellik Uzayı Görselleştirmesi")

    elif os.path.exists("mobilenet_features.npy") and os.path.exists("mobilenet_labels.npy"):
        st.info("Önceden kaydedilmiş PCA/t-SNE verileri bulundu. Aşağıda görselleştirmeyi başlatabilirsiniz.")

        if st.button("📂 Kaydedilmiş Verilerle Görselleştir"):
            X = np.load("mobilenet_features.npy")
            y = np.load("mobilenet_labels.npy")

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
            X_tsne = tsne.fit_transform(X)

            def plot_embedding(X_embedded, title):
                fig, ax = plt.subplots()
                unique_labels = np.unique(y)
                colors = plt.cm.get_cmap("tab10", len(unique_labels))

                for i, label in enumerate(unique_labels):
                    indices = np.where(y == label)
                    ax.scatter(X_embedded[indices, 0], X_embedded[indices, 1],
                               label=label, alpha=0.7, color=colors(i))
                ax.set_title(title)
                ax.set_xlabel("Bileşen 1")
                ax.set_ylabel("Bileşen 2")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            plot_embedding(X_pca, "📊 PCA ile Özellik Uzayı Görselleştirmesi")
            plot_embedding(X_tsne, "🌐 t-SNE ile Özellik Uzayı Görselleştirmesi")


# 5. RAPOR & MATRİSLER
elif menu == "📋 Rapor & Matrisler":
    st.subheader("📋 Sınıflandırma Raporları ve Karışıklık Matrisleri")
    st.write("Aşağıda her modelin test verisindeki tahminlerine karşılık gelen karışıklık matrislerini görebilirsiniz:")

    model_matrix_images = {
        "SVM (94.60%)": "confusion_matrices/SVM_94.60_confusion_matrix.png",
        "Random Forest (85.54%)": "confusion_matrices/RandomForest_85.54_confusion_matrix.png",
        "MobileNet (76.88%)": "confusion_matrices/MobileNet_76.88_confusion_matrix.png",
        "VGG16 (72.21%)": "confusion_matrices/VGG16_72.21_confusion_matrix.png",
        "CNN (70.65%)": "confusion_matrices/CNNBaseline_70.65_confusion_matrix.png"
    }

    for model_name, image_path in model_matrix_images.items():
        st.markdown(f"### 📊 {model_name}")
        st.image(image_path, use_column_width=True)
        st.markdown("---")

        import os
import gdown

def download_if_missing(file_id, output):
    if not os.path.exists(output):
        print(f"{output} indiriliyor...")
        gdown.download(id=file_id, output=output, quiet=False)

download_if_missing("1PUysW4CWS69HnAOTZTkENOhdi56Bm7dt", "svm_model.pkl")
download_if_missing("1CDPS2wtC8BTFeCWQQn5htIi9lms4ROk9", "mobilenet_brain_tumor_model.h5")
download_if_missing("18aoVfmxPaLGV902UulBhQKNrqHEt1TVt", "vgg16_brain_tumor_model.h5")
download_if_missing("1fSVLr3PjR7YsUpBKgU4MvqxG9YzvuN6k", "brain_tumor_cnn_model.h5")



