# Modeli eğit, test et ve tahmin yap
# y_true ve y_pred dizilerini kaydet

import joblib  # Tahmin ve gerçek değerleri dosyaya kaydetmek için

# ... Model eğitimi ve tahmin kodları

# Tahmin ve gerçek etiketler
y_true = [...]  
y_pred = [...]

# Dosyaya kaydet
joblib.dump(y_true, 'y_true.pkl')
joblib.dump(y_pred, 'y_pred.pkl')
