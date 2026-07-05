import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Konfigurasi
MODEL_PATH = 'model_deteksi_sampah_kampus.h5' 
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

print("="*50)
print(f"Mencoba memuat model: {MODEL_PATH}...")
print("="*50)

try:
    model = load_model(MODEL_PATH)
    print("[BERHASIL] Otak AI siap digunakan!\n")
except FileNotFoundError:
    print(f"[GAGAL] File '{MODEL_PATH}' tidak ditemukan!")
    exit()

def uji_banyak_gambar(list_gambar):
    """
    Fungsi untuk memproses sekumpulan gambar secara berurutan.
    """
    for path_gambar in list_gambar:
        if not os.path.exists(path_gambar):
            print(f"\n[PERINGATAN] Gambar '{path_gambar}' tidak ditemukan. Lewati...")
            continue
        
        print(f"\nMenganalisis gambar: {path_gambar}...")
        
        # Preprocessing
        img = image.load_img(path_gambar, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0 
        
        # Prediksi
        predictions = model.predict(x, verbose=0) # verbose=0 agar terminal lebih bersih
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_LABELS[predicted_index]
        confidence = predictions[0][predicted_index] * 100
        
        # Cetak Hasil Terminal
        print("-" * 40)
        print(f" HASIL DETEKSI : {predicted_class.upper()}")
        print(f" CONFIDENCE    : {confidence:.2f}%")
        print("-" * 40)
        
        # Tampilkan Gambar
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Prediksi: {predicted_class.upper()} ({confidence:.2f}%)", fontweight='bold')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    GAMBAR_TEST = [
        "metal_002.jpg", 
        "paper_025.jpg", 
        "plastic_019.jpg"
    ] 
    
    uji_banyak_gambar(GAMBAR_TEST)