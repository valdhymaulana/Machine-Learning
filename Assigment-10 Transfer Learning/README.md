# ♻️ Trash Classification using Transfer Learning (VGG16)

Proyek ini dikembangkan untuk mendeteksi dan mengklasifikasikan jenis sampah di lingkungan kampus Politeknik Negeri Batam. Sistem ini memanfaatkan arsitektur Deep Learning dengan pendekatan Transfer Learning untuk membantu otomatisasi pemilahan sampah.

## 🧠 Mengapa Memilih Transfer Learning?

Pendekatan *Traditional Deep Learning* memiliki beberapa permasalahan utama, yaitu membutuhkan dataset yang sangat besar hingga jutaan sampel, waktu *training* yang sangat lama, komputasi yang mahal dengan GPU *high-end*, dan risiko *overfitting* yang tinggi pada dataset berukuran kecil. Sebagai ilustrasi, melatih model dari awal dengan dataset ImageNet yang memiliki 14 juta gambar bisa memakan waktu 1000+ jam dan menghabiskan biaya komputasi di atas $10,000.

Solusinya, Transfer Learning memungkinkan kita memanfaatkan model yang sudah dilatih sebelumnya untuk mengatasi keterbatasan ini. Transfer Learning adalah metode dalam *machine learning* yang memanfaatkan pengetahuan yang diperoleh dari satu tugas (*source domain*) untuk meningkatkan pembelajaran pada tugas yang berbeda namun terkait (*target domain*).

Dalam arsitektur proyek ini:
* **Source Domain**: Kita menggunakan arsitektur VGG16 yang telah mempelajari representasi fitur (*feature representations*) dari dataset masif ImageNet pada *task* klasifikasi objek secara umum.
* **Target Domain**: Model pre-trained tersebut kita sesuaikan untuk *task* khusus mengenali fitur spesifik pada dataset gambar jenis-jenis sampah (*TrashType*).

## ⚙️ Metodologi & Penanganan Overfitting

Proses pelatihan dieksekusi menggunakan teknik dari Transfer Learning untuk memastikan model belajar dengan optimal dan tidak sekadar menghafal (*overfitting*):

1. **Fixed Feature Extractor (Tahap 1):** Arsitektur model yang lama (*Source Domain*) digunakan murni sebagai *feature extractor*, di mana seluruh layer konvolusinya dibekukan (*freeze pre-trained layers*) agar bobot pengetahuannya tidak rusak. Kemudian, bagian *classifier* (bagian akhir *neural networks*) dimodifikasi dan dilatih ulang untuk mengenali 6 kelas sampah baru.
2. **Handling Overfitting:** Untuk mencegah model menghafal dataset latih yang ukurannya terbatas, mengimplementasikan *Data Augmentation* (manipulasi rotasi, pergeseran letak, dan *horizontal flip*). Implementasi augmentasi untuk memperkaya variasi data ini merujuk pada prinsip riset klasifikasi visual yang dilakukan oleh Naurah Nazhifah, Faturahman Yudanto, Muhamad Fahmi, dan Afiahayati.  juga menginjeksi *layer* **Dropout (0.5)** untuk mematikan 50% neuron secara acak pada *classifier*.
3. **Fine-Tuning (Tahap 2):** Metode ini mirip dengan teknik *Fixed Feature Extractor*, namun pada tahap ini status *frozen* tidak digunakan secara menyeluruh. Membuka kembali 4 *layer* konvolusi terakhir pada model saat proses pelatihan dengan *target domain*. Tujuannya agar representasi fitur tingkat tinggi bisa beradaptasi secara presisi dengan bentuk material seperti logam, kaca, dan kardus.

# 📊 Hasil Pengujian
Berikut adalah bukti ketangguhan model saat memproses visual objek sampah dunia nyata yang belum pernah dilihat pada data training:

# Pengujian 1: Kaleng Logam (Metal)

<img width="1095" height="272" alt="image" src="https://github.com/user-attachments/assets/857a0469-6788-4e9f-9929-fa77887b414e" />

Model berhasil mengekstraksi fitur pantulan cahaya dari material logam dan mengklasifikasikannya sebagai METAL dengan confidence tinggi (95.99%).

# Pengujian 2: Kertas (Paper)

<img width="889" height="306" alt="image" src="https://github.com/user-attachments/assets/5f6be756-e555-4b20-b7c6-2f6518aeef59" />

Mengklasifikasikannya sebagai Paper dengan tingkat keyakinan 80.60%.

# Pengujian 3: Botol Plastik (Plastic)

<img width="780" height="337" alt="image" src="https://github.com/user-attachments/assets/721a4563-1bf4-4d10-8c4d-7d4974ee1d07" />

Model sukses mengidentifikasi lekukan dan karakteristik transparansi material plastik pada objek, mengklasifikasikannya sebagai PLASTIC dengan confidence 98.10%.

# Untuk model dan dataset dapat dilihat pada link drive di bawah
https://drive.google.com/drive/folders/1EVkLl8Vvy8vJuTCUbU1nL5SbIPj8S6Lb?usp=drive_link
