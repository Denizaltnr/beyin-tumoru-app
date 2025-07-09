import matplotlib.pyplot as plt

# Modeller ve doğruluk oranları
models = ['CNN', 'MobileNet', 'VGG16', 'SVM', 'Random Forest']
accuracies = [70.65, 76.88, 72.21, 94.60, 85.54]

plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'purple', 'orange', 'red'])

plt.ylim(0, 100)
plt.ylabel('Doğruluk (%)')
plt.title('Model Doğruluk Oranları Karşılaştırması')

# Çubukların üstüne yüzde değerleri yazma
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height - 7, f'{height:.2f}%', 
             ha='center', color='white', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
