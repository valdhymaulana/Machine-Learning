# Iris MLP Activation Comparison

## Deskripsi Tugas
Tugas ini membandingkan performa Multi-Layer Perceptron (MLP) pada dataset Iris dengan tiga fungsi aktivasi berbeda:
- Sigmoid
- Tanh
- ReLU

Dataset yang digunakan adalah `Iris.csv`, diambil dari Kaggle.

## Tujuan
- Menilai performa MLP menggunakan fungsi aktivasi yang berbeda.
- Mengukur akurasi, waktu pelatihan, dan jumlah iterasi untuk setiap aktivasi.
- Menentukan fungsi aktivasi terbaik untuk dataset Iris.

## Metode
1. Data dibaca dari file `Iris.csv`.
2. Kolom `Id` dihapus jika ada karena tidak berpengaruh pada pelatihan.
3. Fitur dan label dipisah: fitur `sepal length`, `sepal width`, `petal length`, `petal width`; label `Species`.
4. Data dibagi menjadi 80% train dan 20% test secara acak dengan `random_state=42`.
5. Fitur diskalakan menggunakan `StandardScaler`.
6. Model MLP dilatih dengan `MLPClassifier` dan arsitektur `hidden_layer_sizes=(10, 10)`.
7. Evaluasi dilakukan pada data test.

## Output
- Tabel perbandingan fungsi aktivasi:
  - Akurasi (%)
  - Waktu pelatihan (detik)
  - Jumlah iterasi sampai konvergensi
- Ringkasan hasil terbaik.

## Hasil
- Ketiga fungsi aktivasi: Sigmoid, Tanh, dan ReLU memberikan akurasi yang sama pada dataset Iris.
- ReLU biasanya menjadi pilihan terbaik karena waktu pelatihan paling cepat dan jumlah iterasi paling sedikit.

## Kesimpulan
Fungsi aktivasi terbaik untuk dataset Iris dalam tugas ini adalah **ReLU**, karena memberikan kecepatan pelatihan terbaik dengan akurasi setara dibandingkan Sigmoid dan Tanh.

