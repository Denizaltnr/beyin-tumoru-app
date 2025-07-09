from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "", 14)
        self.cell(0, 10, "Beyin Tümörü Sınıflandırma Teknik Raporu", align="C", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}", align="C")

# 🔹 Dosya yolları
font_path = "C:/Users/ben_d/Desktop/VeriProje/fonts/dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf"
bold_font_path = "C:/Users/ben_d/Desktop/VeriProje/fonts/dejavu-sans-ttf-2.37/ttf/DejaVuSans-Bold.ttf"
matrix_folder = "C:/Users/ben_d/Desktop/VeriProje/confusion_matrices"

# 🔹 PDF başlat
pdf = PDF()
pdf.add_font("DejaVu", "", font_path)
pdf.add_font("DejaVu", "B", bold_font_path)
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# 🔹 İçerik başlıkları ve açıklamalar
sections = [
    ("📘 TEKNİK PROJE RAPORU\nProje Başlığı:",
     "Beyin Tümörü Sınıflandırması: Derin Öğrenme ve Makine Öğrenmesi Yöntemleriyle Karşılaştırmalı Bir Yaklaşım"),

    ("1. İş Anlayışı (Business Understanding)",
     "Amaç: MRI görüntüleri üzerinden beyin tümörü sınıflandırması yaparak, doktorlara tanı sürecinde destek olacak bir yapay zeka sistemi geliştirmek.\n\n"
     "Problem Tanımı: Tümör türlerinin manuel olarak teşhisi uzmanlık gerektirir ve zaman alır. Otomatik sınıflandırma sistemleri bu süreci hızlandırabilir ve doğruluğu artırabilir.\n\n"
     "Hedef Kullanıcılar: Radyologlar, sağlık bilişimi uzmanları ve karar destek sistemi entegrasyonuna ihtiyaç duyan sağlık yazılımları."),

    ("2. Veri Anlayışı (Data Understanding)",
     "Veri Kaynağı: Kaggle - “Brain Tumor Classification (MRI)” veri seti.\n\n"
     "Sınıflar:\n- glioma_tumor\n- meningioma_tumor\n- pituitary_tumor\n- no_tumor\n\n"
     "Veri Özellikleri:\n- Görüntü Sayısı: 3264 eğitim, 702 test.\n- Format: RGB JPEG görüntüler.\n- Kaynak: MRI taramaları.\n\n"
     "Gözlemler:\n- Görüntülerin çözünürlükleri farklı.\n- Sınıflar arası dengesizlik orta düzeyde."),

    ("3. Veri Hazırlama (Data Preparation)",
     "Yeniden Boyutlandırma:\n- CNN için: 150x150\n- Transfer learning için: 224x224\n\n"
     "Normalizasyon:\n- CNN için /255,\n- MobileNet ve VGG16 için preprocess_input().\n\n"
     "Etiketleme: Klasör isimlerinden etiketler çıkarıldı.\n\n"
     "Özellik Çıkarımı: MobileNet kullanılarak son katmandan öznitelikler çıkarıldı.\n\n"
     "Veri Ayrımı: Eğitim/Test oranı: 80/20."),

    ("4. Modelleme (Modeling)",
     "Model-Açıklama\n"
     "CNN,Sıfırdan oluşturulmuş, temel Conv2D, MaxPooling, Flatten ve Dense katmanları içeriyor.\n"
     "MobileNet,ImageNet üzerinde önceden eğitilmiş; sınıflandırma katmanları çıkarılıp yeni Dense katmanlar eklendi.\n"
     "VGG16,Transfer learning modeli; tüm katmanlar dondurularak Dense katman eklendi.\n"
     "SVM,tMobileNet ile çıkarılan öznitelikler üzerinden sklearn SVM eğitildi.\n"
     "Random Forest,Aynı öznitelikler ile ağaç tabanlı RF sınıflandırıcısı eğitildi.\n\n"
     "Model Eğitim Detayları:\n"
     "- Optimizasyon: Adam (CNN), Grid Search (SVM/RF için hiperparametre optimizasyonu).\n"
     "- Epoch: CNN 20 epoch, transfer modeller 10 epoch fine-tune edildi.\n"
     "- Kaybın ve doğruluğun izlenmesi için erken durdurma kullanıldı."),

    ("5. Değerlendirme (Evaluation)",
     "Modellerin doğruluk oranları:\n"
     "- CNN: 70.65%\n"
     "- MobileNet: 76.88%\n"
     "- VGG16: 72.21%\n"
     "- SVM: 94.60% (En başarılı model)\n"
     "- Random Forest: 85.54%\n\n"
     "Ek Değerlendirmeler:\n"
     "- Her model için confusion matrix analizleri yapıldı.\n"
     "- Sınıflar arası karışıklıklar incelendi.\n"
     "- SVM modeli hem genel doğrulukta hem sınıflar arası dengede en iyi sonuçları verdi."),

    ("6. Dağıtım (Deployment)",
     "Platform: Streamlit\n\n"
     "Özellikler:\n"
     "- Kullanıcı görsel yükleyebilir.\n"
     "- Tüm modellerle tahmin alınabilir.\n"
     "- PCA ve t-SNE ile öznitelik uzayı görselleştirilebilir.\n"
     "- Model performansları grafiklerle karşılaştırılır.\n\n"
     "Dosya Entegrasyonları:\n"
     "- .pkl model dosyaları\n"
     "- .png formatında confusion matrix görselleri"),

    ("7. Sonuç ve Öneriler (Conclusion & Recommendations)",
     "Sonuç: SVM modeli, MobileNet öznitelikleri ile en başarılı sonucu verdi.\n\n"
     "Güçlü Yanlar:\n"
     "- Transfer learning + klasik ML birleşimi etkili oldu.\n"
     "- Model çeşitliliği sayesinde performans kıyaslaması sağlandı.\n\n"
     "Öneriler:\n"
     "- Veri artırma teknikleriyle eğitim seti genişletilmeli.\n"
     "- Yeni derin öğrenme mimarileri (EfficientNet, ResNet) denenebilir.\n"
     "- Ensemble modeller ile hibrit yaklaşımlar uygulanabilir.")
]

# 🔹 Bölümleri yazdır
for title, content in sections:
    pdf.set_font("DejaVu", "B", 12)
    pdf.multi_cell(0, 8, title)
    pdf.ln(1)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 7, content)
    pdf.ln(4)

# 🔹 8. Karışıklık Matrisleri
pdf.add_page()
confusion_data = {
    "SVM_94.60_confusion_matrix.png": "SVM modeli, tüm sınıflarda dengeli tahminler yapmıştır. Hata oranı düşüktür.",
    "RandomForest_85.54_confusion_matrix.png": "Random Forest bazı sınıflarda sapma göstermiştir ancak genel doğruluk yüksektir.",
    "MobileNet_76.88_confusion_matrix.png": "Özellikle 'no_tumor' sınıfında karışıklık oranı yüksektir.",
    "VGG16_72.21_confusion_matrix.png": "VGG16 sınıflar arasında kısmi dengesizlikler göstermektedir.",
    "CNNBaseline_70.65_confusion_matrix.png": "Temel CNN modeli en düşük doğruluğa sahiptir; özellikle glioma ve no_tumor sınıflarında zayıf kalmıştır."
}

for img_file, desc in confusion_data.items():
    img_path = os.path.join(matrix_folder, img_file)
    if os.path.exists(img_path):
        if pdf.get_y() > 210:
            pdf.add_page()
        pdf.image(img_path, x=25, w=160)
        pdf.ln(4)
        pdf.set_font("DejaVu", "", 10)
        pdf.multi_cell(0, 6, desc)
        pdf.ln(5)
    else:
        print(f"⚠️ Görsel eksik: {img_path}")

# 🔹 PDF Kaydet
output_path = "beyin_tumoru_teknik_rapor.pdf"
pdf.output(output_path)
print("✅ PDF başarıyla oluşturuldu:", output_path)
