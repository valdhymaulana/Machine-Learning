# Wine Quality KMeans Clustering

## Deskripsi Program
- Notebook ini menggunakan dataset `winequality-red.csv` untuk analisis unsupervised clustering.
- Metode yang digunakan adalah KMeans untuk menemukan pola kelompok dalam data wine.
- Visualisasi utama menggunakan `alcohol` vs `quality` sebagai representasi 2D.

## Isi Program
- Import library penting: `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`.
- Baca data CSV `winequality-red.csv` ke DataFrame `df`.
- Lakukan eksplorasi data awal dengan plot scatter, boxplot, histogram, dan `df.describe()`.
- Hapus duplikat data untuk menjaga kualitas dataset.
- Pilih semua fitur numerik kecuali `quality` sebagai input untuk clustering.
- Standarisasi fitur menggunakan `StandardScaler`.

## Cara Kerja
1. Data dimuat dan diperiksa struktur kolom.
2. EDA dilakukan untuk melihat sebaran dan anomali data.
3. Duplikat dihapus agar tidak mempengaruhi hasil clustering.
4. Fitur dinormalisasi agar semua variabel masuk dalam skala yang seimbang.
5. KMeans dijalankan dengan dua pendekatan:
   - Elbow Method untuk memilih jumlah cluster terbaik.
   - Via-Score Plot (`yellowbrick`) untuk memvalidasi pilihan jumlah cluster.
6. Hasil cluster ditambahkan ke DataFrame dalam kolom `cluster_elbow` dan `cluster_via`.
7. Visualisasi membandingkan hasil cluster dengan label `quality` asli.

## Catatan
- `quality` pada dataset adalah label numerik yang merepresentasikan nilai kualitas wine.
- KMeans clustering tidak menggunakan label `quality` sebagai target, tetapi sebagai referensi pembandingan.
- Plot `alcohol vs quality` hanya untuk visualisasi; model clustering bekerja pada semua fitur numerik.

## Tujuan
- Memahami cara kerja clustering KMeans pada dataset wine.
- Menentukan apakah ada pola kelompok alami dalam data tanpa label target eksplisit.
- Mengilustrasikan hasil cluster dengan perbandingan ke kualitas wine.
