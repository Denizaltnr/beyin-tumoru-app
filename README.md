# 🧠 Brain Tumor Classification Web App

Bu proje, beyin tümörü sınıflandırması yapan bir yapay zeka uygulamasıdır. Derin öğrenme ve makine öğrenmesi modelleri kullanılarak geliştirilmiştir.

## 🚀 Özellikler

- CNN, VGG16, MobileNet, SVM ve Random Forest modelleriyle sınıflandırma
- Streamlit ile kolay kullanım arayüzü
- Google Drive üzerinden otomatik model indirme desteği

## 📁 Kurulum

1. Depoyu klonla:

```bash
git clone https://github.com/kullanici_adin/beyin-tumoru-app.git
cd beyin-tumoru-app

2.Gerekli paketleri yükle:

pip install -r requirements.txt
pip install gdown

3.Model dosyalarını indir:

python download_models.py

4.Uygulamayı başlat:

streamlit run app.py