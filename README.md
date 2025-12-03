🚑 AI-Powered Smart Traffic Management System for Emergency Vehicles

Acknowledgement: This project is supported by TÜBİTAK (The Scientific and Technological Research Council of Türkiye) under the 2209-A University Students Research Projects Support Program.

📖 Project Abstract

This research project aims to minimize the response time of emergency vehicles (ambulances) in heavy traffic. We developed a Hybrid Detection System prototype running on a Raspberry Pi 5 that combines Acoustic Signal Processing and Computer Vision.

Unlike traditional systems that rely solely on cameras (which fail in blind spots) or GPS (which has latency), our system utilizes MFCC (Mel-Frequency Cepstral Coefficients) features to detect siren sounds and verifies them with visual object detection to dynamically control traffic lights.

⚙️ System Architecture (Sensor Fusion)

The core innovation of this project is the Data Fusion of audio and visual inputs to prevent false positives.

graph TD
    subgraph "Sensory Input Layer"
        Mic[Acoustic Sensors] -->|Raw Audio Stream| PreProcess[Noise Cancellation & Normalization]
        Cam[Camera Feed] -->|Video Stream| VisionInput[Frame Capture]
    end

    subgraph "Processing Unit (Raspberry Pi 5)"
        PreProcess -->|Extract Features| MFCC[MFCC Feature Extraction]
        MFCC -->|Input| AudioAI[Audio CNN Model]
        
        VisionInput -->|Input| YOLO[Object Detection Model (TensorFlow)]
        
        AudioAI -- "Siren Detected? (Probability > 0.8)" --> FusionEngine{Decision Fusion Engine}
        YOLO -- "Ambulance Visual Confirmed?" --> FusionEngine
    end

    subgraph "Actuation Layer"
        FusionEngine -->|YES: Priority Mode| Controller[Traffic Light Controller]
        Controller -->|Switch to Green| TrafficLights[Physical Traffic Lights]
        FusionEngine -->|NO: Standard Cycle| Standard[Normal Traffic Flow]
    end


🛠️ Technology Stack & Methods

🧠 Artificial Intelligence & Signal Processing

Audio Processing: Librosa for extracting MFCC features from raw audio waves.

Deep Learning (Audio): Custom Convolutional Neural Network (CNN) trained on siren vs. noise datasets using TensorFlow/Keras.

Computer Vision: OpenCV and TensorFlow Lite for real-time vehicle classification.

Data Fusion: Custom algorithm to weight audio and video probabilities for final decision making.

hardware

Core Unit: Raspberry Pi 5 (Selected for high processing power for concurrent AI models).

Sensors: High-fidelity USB Microphones (for acoustic pickup).

Vision: Raspberry Pi Camera Module / USB Webcam.

🔬 Methodology Details

1. Acoustic Detection (The "Ear")

We don't just look for loud noises; we look for the specific signature of a siren.

Feature Extraction: We use Mel-Frequency Cepstral Coefficients (MFCC) to convert audio waves into a visual representation (spectrogram) that the AI can understand.

Noise Filtering: Applied spectral gating to remove environmental city noise (horns, wind).

2. Visual Verification (The "Eye")

Once a potential siren is heard, the vision module activates to confirm the presence of an emergency vehicle, saving computational power and preventing false alarms from other loud sources.

📊 Key Goals & Metrics

Accuracy: Target >95% accuracy in siren classification.

Latency: Inference time <1 second on edge hardware.

Impact: Estimated 20% reduction in ambulance travel time through busy intersections.

📂 Project Structure

AI_Traffic_System/
├── data/
│   ├── audio/          # Raw audio samples (Siren/Noise)
│   ├── features/       # Extracted MFCC features (.npy files)
│   └── models/         # Trained .h5 and .tflite models
├── src/
│   ├── sound_detector.py   # Audio processing & inference logic
│   ├── vision_detector.py  # Object detection logic
│   └── traffic_control.py  # GPIO control for traffic lights
├── notebooks/          # Jupyter notebooks for model training
├── requirements.txt    # Dependencies (librosa, tensorflow, opencv-python)
└── main.py             # Main execution loop


👥 Team

Developers: Kemal Atalar, Beyza Kuzu

Supervisor: Doç. Dr. Ferhat Uçar

Institution: Fırat University, Software Engineering Department

This project is part of the undergraduate research curriculum.
