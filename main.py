# main.py
# data/audio klasöründeki tüm ses dosyalarını analiz eder.

import os
from sound_detector import load_audio, simple_siren_rule

AUDIO_DIR = "data/audio"

def scan_audio_folder(folder=AUDIO_DIR):
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".wav", ".flac", ".mp3")):
            continue
        path = os.path.join(folder, fname)
        y, sr = load_audio(path)
        is_siren, stats = simple_siren_rule(y, sr)
        print(f"{fname} -> Siren: {is_siren}, stats: {stats}")

if __name__ == "__main__":
    scan_audio_folder()
