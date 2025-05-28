import os
import streamlit as st

if not os.path.exists("svm_model.pkl"):
    import download_models

st.title("🧠 Beyin Tümörü Sınıflandırma Uygulaması")
st.write("Model dosyaları başarıyla yüklendiyse, test görselinizi yükleyerek tahmin alabilirsiniz.")
