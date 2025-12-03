# 🚦 AI Traffic System

Yapay zeka tabanlı trafik sistemi - Ses analizi ile siren ve gürültü tespiti

## 📋 Proje Açıklaması

Bu proje, trafik ortamındaki sesleri analiz ederek siren sesleri ve diğer gürültüleri ayırt edebilen bir yapay zeka sistemidir. Makine öğrenmesi teknikleri kullanarak ses sinyallerini sınıflandırır.

## 🔧 Kullanılan Teknolojiler

- **Python 3.x**
- **TensorFlow/Keras** - Derin öğrenme
- **Librosa** - Ses işleme
- **NumPy** - Sayısal hesaplamalar
- **Scikit-learn** - Makine öğrenmesi araçları

## 📁 Proje Yapısı

```
AI_Traffic_System/
├── data/
│   ├── audio/          # Ses dosyaları (siren, gürültü)
│   └── video/          # Video dosyaları (gelecek özellik)
├── models/             # Eğitilmiş AI modelleri
├── main.py            # Ana uygulama
├── sound_detector.py  # Ses analizi ve AI modeli
└── vision_detector.py # Görüntü işleme (placeholder)
```

## 🚀 Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Ses dosyalarınızı `data/audio/` klasörüne yerleştirin
   - Siren sesleri: `siren_*.wav` formatında
   - Gürültü sesleri: `gurultu_*.wav` formatında

## 💻 Kullanım

### Model Eğitimi
```bash
python sound_detector.py
```

### Ses Analizi
```bash
python main.py
```

## 🎯 Özellikler

- ✅ MFCC özellik çıkarma
- ✅ Derin öğrenme ile sınıflandırma
- ✅ Toplu ses dosyası analizi
- ✅ Model kaydetme/yükleme
- 🔄 Görüntü işleme (geliştirme aşamasında)

## 📊 Model Detayları

- **Giriş**: 40 MFCC özelliği
- **Mimari**: 3 katmanlı tam bağlı sinir ağı
- **Aktivasyon**: ReLU (gizli katmanlar), Softmax (çıkış)
- **Regularization**: Dropout (0.5)
- **Optimizer**: Adam

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Proje hakkında sorularınız için issue açabilirsiniz.