# 🌸 Klasifikasi Spesies Bunga Iris dengan KNN

Proyek *Machine Learning* ini mengklasifikasikan spesies bunga Iris (**Setosa, Versicolor, Virginica**) berdasarkan dimensi fisik kelopaknya. Model prediksi ini dibangun menggunakan algoritma **K-Nearest Neighbors (KNN)** yang dioptimasi secara komprehensif melalui tahapan `GridSearchCV`.

Seluruh *pipeline* pemrosesan data telah diterapkan secara terstruktur untuk menghasilkan model prediksi dengan tingkat akurasi tinggi.

## ⚙️ Pipeline Proyek

Alur kerja yang diterapkan dalam repositori ini meliputi:
- **Exploratory Data Analysis (EDA):** Eksplorasi visual sebaran dan korelasi ukuran fisik bunga.
- **Pembersihan Data:** Memastikan dataset bebas dari duplikasi dan anomali.
- **Feature Scaling:** Standardisasi rentang ukuran menggunakan agar perhitungan metrik spasial berjalan optimal.
- **Optimasi Model:** Pencarian *hyperparameter* terbaik (Hyperparameter Tuning) secara otomatis menggunakan `GridSearchCV`.
- **Evaluasi Matriks:** Validasi performa model menggunakan matriks evaluasi yang komprehensif (*Accuracy, Precision, Recall, F1-Score*).
