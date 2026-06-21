# ♻️ Trash Type Image Classification (CNN)

Repositori ini berisi implementasi algoritma **Convolutional Neural Network (CNN)** untuk mendeteksi dan mengklasifikasikan 6 jenis sampah (Kardus, Kaca, Logam, Kertas, Plastik, Residu). Proyek ini dikembangkan menggunakan kerangka kerja TensorFlow/Keras.

Repositori ini memuat dua program utama: program untuk melatih "otak" AI (`train.py`) dan program untuk menguji kepintaran AI tersebut menggunakan gambar baru (`tes.py`).

---

## ⚙️ 1. Cara Kerja Program Training (`train.py`)

Program ini bertanggung jawab untuk membaca ribuan gambar sampah mentah, mengekstraksi fiturnya, dan melatih jaringan saraf tiruan. Berikut adalah rincian tahapan kerjanya:

### A. Data Preprocessing & Augmentation
Karena AI membutuhkan standarisasi, gambar tidak bisa langsung dimasukkan mentah-mentah.
* **`ImageDataGenerator`**: Kelas ini digunakan untuk memuat gambar dari folder.
* **`rescale=1./255`**: Mengubah rentang nilai piksel gambar asli (0-255) menjadi nilai desimal (0-1). Proses normalisasi ini wajib agar model CNN lebih mudah dan cepat melakukan kalkulasi matriks.
* **Augmentasi**: Gambar diputar (`rotation_range=20`), di-zoom (`zoom_range=0.2`), dan dibalik secara horizontal (`horizontal_flip=True`). Ini membuat AI belajar bahwa botol yang miring atau terbalik tetaplah sebuah botol, sehingga mencegah *overfitting* (model hanya menghafal data).
* **`validation_split=0.2`**: Membelah dataset secara otomatis; 80% data digunakan untuk belajar (*training*), dan 20% sisanya disembunyikan untuk ujian evaluasi (*validation*).

### B. Arsitektur Jaringan (CNN Building)
Model dibangun menggunakan struktur `Sequential` (berurutan) yang dibagi menjadi dua fase utama:

**Fase 1: Feature Extraction (Ekstraksi Fitur)**
* **Layer Conv2D Pertama (32 filter)**: Bertugas mencari pola dasar pada gambar, seperti tepi, garis lurus, atau sudut. Menggunakan ukuran filter (kernel) $3\times3$ dan aktivasi `ReLU` (mengubah nilai negatif menjadi 0).
* **Layer MaxPooling2D (2x2)**: Meringkas gambar dengan hanya mengambil nilai piksel tertinggi di setiap area $2\times2$. Tujuannya untuk memperkecil ukuran matriks tanpa menghilangkan informasi spasial penting.
* **Layer Conv2D Kedua (64 filter) + MaxPooling**: Menggabungkan garis dan tepi dari layer pertama menjadi bentuk yang lebih kompleks (misal: lengkungan botol atau tekstur kusut kertas).

**Fase 2: Classification (Klasifikasi)**
* **`Flatten()`**: Meratakan matriks 3D yang dihasilkan dari fase ekstraksi fitur menjadi sebuah susunan vektor 1D (satu dimensi) panjang.
* **`Dense(128)`**: Sebuah *Hidden Layer* (Multi Layer Perceptron) dengan 128 neuron yang saling terhubung untuk mempelajari kombinasi fitur.
* **Output Layer `Dense(num_classes, activation='softmax')`**: Layer terakhir dengan 6 neuron (sesuai jumlah kategori sampah). Fungsi `softmax` menekan *output* dari keenam neuron tersebut agar menjadi nilai probabilitas (persentase) yang jika dijumlahkan totalnya 100%.

### C. Kompilasi & Eksekusi
* Model dikompilasi menggunakan *optimizer* `Adam` dan metode perhitungan kerugian (*loss function*) `categorical_crossentropy`.
* Pelatihan dijalankan selama 10 putaran (`epochs=10`).
* Setelah selesai belajar, matriks bobot "otak" AI tersebut diamankan ke dalam penyimpanan lokal dengan perintah `model.save('model_sampah_kampus.keras')` agar tidak perlu dilatih ulang di masa depan.

---

## 🔍 2. Cara Kerja Program Pengujian (`tes.py`)

Setelah model memiliki matriks pengetahuan (`.keras`), program kedua ini dibuat khusus untuk melakukan *Inference* (prediksi) terhadap objek gambar di dunia nyata.

### A. Memuat Model & Kelas
* **`load_model()`**: Membangkitkan kembali arsitektur dan matriks bobot model CNN dari file `model_sampah_kampus.keras` langsung ke dalam memori komputer.
* **Daftar Kelas**: Nama kelas sampah didefinisikan secara manual ke dalam urutan alfabet (*array*) agar indeks probabilitas nanti bisa dikonversi menjadi teks yang bisa dibaca manusia.

### B. Pra-pemrosesan Gambar Uji (Wajib Sama dengan Training)
Gambar baru (misalnya `tes_sampah.jpg`) tidak bisa langsung ditebak. Gambar harus melewati jalur perlakuan yang sama persis seperti saat *training*:
1. Di-*resize* paksa menjadi ukuran `128x128` piksel.
2. Diubah menjadi matriks angka menggunakan `img_to_array`.
3. Ditambahkan satu dimensi palsu di depannya menggunakan `np.expand_dims` (karena Keras selalu menuntut format *batch*, jadi formatnya berubah dari `(128, 128, 3)` menjadi `(1, 128, 128, 3)`).
4. Dinormalisasi pikselnya dengan membaginya menggunakan `255.0`.

### C. Eksekusi Prediksi (*Feedforward*)
* **`model.predict()`**: Gambar yang sudah berwujud tensor matriks dimasukkan ke dalam CNN. Model akan melakukan operasi perkalian matriks dari ujung ke ujung hingga menghasilkan 6 nilai probabilitas di *output layer*.
* **`np.argmax()`**: Fungsi NumPy ini bertugas mencari indeks atau posisi angka probabilitas terbesar dari ke-6 hasil tersebut. Jika indeks terbesar ada di posisi ke-3, maka program akan mencocokkannya dengan *array* `kelas_sampah` di indeks ke-3 (yaitu 'Kertas').
* Hasil tebakan akhir dan persentase keyakinan model dicetak ke terminal.

---

## 📸 3. Hasil Uji Coba

Sistem telah diuji menggunakan gambar eksternal yang belum pernah dilihat oleh AI pada saat *training*.

**Objek Gambar yang Diuji:**
<img width="530" height="400" alt="image" src="https://github.com/user-attachments/assets/23933326-b7dd-44ec-a196-7da85f8281dd" />

**Karena keterbatasn ukuran file, maka model hasil latih dapat dilihat pada drive di bawah:**
https://drive.google.com/drive/folders/1VR-ueKv_mLSgBsBM7BnCrVeVeDFrMNIb?usp=drive_link

**Hasil Analisis Sistem (Terminal Log):**
<img width="780" height="183" alt="image" src="https://github.com/user-attachments/assets/8ae4a2d5-123b-4059-becb-61cb3f97393c" />


**Kesimpulan Pengujian:**
Walaupun objek kertas majalah tersebut memiliki warna kontras tinggi dan visual wajah manusia (yang dapat membingungkan filter deteksi pola), algoritma CNN berhasil melakukan abstraksi fitur dengan sangat baik. Program mengeksekusi ekstraksi secara akurat dan memprediksi objek tersebut sebagai **KERTAS** dengan tingkat keyakinan (*confidence*) sebesar **80.54%**.

---

