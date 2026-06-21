import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# 1. Konfigurasi File
MODEL_PATH = 'model_sampah_kampus.h5' 
GAMBAR_TES = 'tes_sampah.jpg' # Ganti sesuai nama file foto Anda

# 2. Pemetaan Kelas (Urutannya HARUS sesuai abjad nama folder saat di Colab)
# ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
kelas_sampah = ['Kardus', 'Kaca', 'Logam', 'Kertas', 'Plastik', 'Residu/Campuran']

def eksekusi_prediksi():
    # Cek ketersediaan file
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' tidak ditemukan!")
        return
    if not os.path.exists(GAMBAR_TES):
        print(f"Error: Gambar '{GAMBAR_TES}' tidak ditemukan!")
        return

    print("Memuat model CNN ke dalam memori, mohon tunggu...\n")
    model = load_model(MODEL_PATH)

    # 3. Preprocessing (Mempersiapkan gambar agar mirip dengan data training)
    img = image.load_img(GAMBAR_TES, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Menambah dimensi untuk batch
    img_array /= 255.0  # WAJIB: Rescaling yang sama dengan ImageDataGenerator

    # 4. Eksekusi Jaringan Feedforward
    prediksi = model.predict(img_array)
    
    # Karena output layer pakai aktivasi 'softmax', hasilnya adalah matriks probabilitas
    index_tertinggi = np.argmax(prediksi[0])
    kategori_hasil = kelas_sampah[index_tertinggi]
    tingkat_keyakinan = np.max(prediksi[0]) * 100

    # 5. Tampilkan Hasil
    print("=" * 40)
    print(" HASIL ANALISIS KLASIFIKASI SAMPAH")
    print("=" * 40)
    print(f"Gambar Diprediksi Sebagai : {kategori_hasil.upper()}")
    print(f"Tingkat Keyakinan Model   : {tingkat_keyakinan:.2f}%")
    print("=" * 40)

if __name__ == '__main__':
    eksekusi_prediksi()