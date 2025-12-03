# sound_detector.py

import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. VERİ SETİNİ HAZIRLAMA
# data klasörünün yolu. Bu yolun doğru olduğundan emin olun.
DATA_PATH = "data/audio"
# Modeli kaydedeceğimiz yol.
MODEL_PATH = "models/sound_model.h5"

# Etiketler ve ses dosyası yollarını saklayacağımız listeler
labels = []
filepaths = []

# data klasöründeki her bir dosyayı gez
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".wav"):
        filepath = os.path.join(DATA_PATH, filename)
        
        # Dosya isminden etiketi çıkar. 
        # Örnek dosya isimleri: "siren_1.wav", "gurultu_1.wav" olmalı
        label = filename.split('_')[0]
        
        filepaths.append(filepath)
        labels.append(label)

print(f"Toplam {len(filepaths)} adet ses dosyası bulundu.")

# Etiketleri (siren, gurultu gibi metinleri) sayılara dönüştür (0, 1 gibi)
le = LabelEncoder()
y = le.fit_transform(labels)
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(le.classes_))


# 2. SES ÖZELLİKLERİNİ ÇIKARMA (FEATURE EXTRACTION - MFCC)
all_features = []
for filepath in filepaths:
    try:
        # Sesi yükle. `sr=None` orijinal örnekleme oranını korur.
        # `sr=22050` gibi bir değere sabitleyerek tüm sesleri standartlaştırabiliriz.
        samples, sample_rate = librosa.load(filepath, sr=22050)
        
        # Ses çok kısaysa veya boşsa atla
        if len(samples) < 4096:
            print(f"Uyarı: {filepath} dosyası çok kısa, atlanıyor.")
            continue

        # MFCC özelliklerini çıkar
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        all_features.append(mfcc_scaled)
        
    except Exception as e:
        print(f"Hata: {filepath} dosyası işlenemedi. Hata mesajı: {e}")

# Özellik listesini numpy dizisine çevir
X = np.array(all_features)

print(f"Özellik matrisinin boyutu: {X.shape}")
print(f"Etiket matrisinin boyutu: {y_one_hot.shape}")

# 3. VERİ SETİNİ EĞİTİM VE TEST OLARAK AYIRMA
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# 4. YAPAY ZEKA MODELİNİ OLUŞTURMA (KERAS ile)
model = tf.keras.models.Sequential([
    # Giriş katmanı: 40 MFCC özelliğimiz olduğu için input_shape=(40,)
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.5), # Aşırı öğrenmeyi (overfitting) engellemek için
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Çıkış katmanı: İki sınıfımız var (siren, gurultu), bu yüzden 2 nöron
    # 'softmax' aktivasyonu, çıktıyı olasılık olarak verir.
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. MODELİ EĞİTME
print("\nModel Eğitimi Başlıyor...")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
print("Model Eğitimi Tamamlandı.")

# 6. MODELİ DEĞERLENDİRME
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Doğruluğu (Accuracy): {test_acc:.4f}")

# 7. EĞİTİLMİŞ MODELİ KAYDETME
model.save(MODEL_PATH)
print(f"\nEğitilmiş model '{MODEL_PATH}' olarak kaydedildi.")