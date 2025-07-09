import os
import matplotlib.pyplot as plt

data_dir = r'C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI\Training'
class_names = sorted(os.listdir(data_dir))

counts = []
for c in class_names:
    path = os.path.join(data_dir, c)
    counts.append(len(os.listdir(path)))

plt.figure(figsize=(8,5))
plt.bar(class_names, counts, color='skyblue')
plt.title('Training Veri Seti Sınıf Dağılımı')
plt.xlabel('Sınıflar')
plt.ylabel('Görüntü Sayısı')
plt.grid(axis='y')
plt.show()
