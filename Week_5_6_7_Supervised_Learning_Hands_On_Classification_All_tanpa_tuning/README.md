### Ringkasan Model Naive Bayes

Bagian ini mengimplementasikan model klasifikasi **Naive Bayes** untuk memprediksi kelangsungan hidup penumpang Titanic. Model ini dipilih karena kesederhanaan dan efisiensinya dalam menangani tugas klasifikasi, terutama dengan fitur-fitur yang berdistribusi seperti Gaussian.

#### Langkah-langkah Implementasi:

1.  **Persiapan Data:**
    *   Data `train` (setelah proses Feature Engineering seperti imputasi `Age`, penghapusan kolom `Cabin`, `PassengerId`, `Name`, `Ticket`, serta encoding `Sex` dan `Embarked`) digunakan sebagai input.
    *   Data kemudian dibagi menjadi `X_train`, `X_test`, `y_train`, dan `y_test` untuk pelatihan dan pengujian model.

2.  **Inisialisasi & Pelatihan Model:**
    *   Model `GaussianNB` dari `sklearn.naive_bayes` diinisialisasi.
    *   Model dilatih (`nb.fit(X_train, y_train)`) menggunakan data pelatihan.
    *   Prediksi (`y_pred_nb = nb.predict(X_test)`) dilakukan pada data uji.

3.  **Evaluasi Model:**
    *   Kinerja model Naive Bayes dievaluasi menggunakan metrik standar:
        *   **Akurasi:** Tingkat prediksi yang benar secara keseluruhan.
        *   **Confusion Matrix:** Detail True Positives, True Negatives, False Positives, dan False Negatives.
        *   **Classification Report:** Menampilkan Precision, Recall, dan F1-Score untuk setiap kelas (selamat/tidak selamat).
    *   **Hasil Evaluasi Naive Bayes:**
        *   Akurasi Model Naive Bayes: **0.78**
        *   Confusion Matrix:
            ```
            [[135  32]
             [ 26  74]]
            ```
        *   Classification Report menunjukkan performa yang seimbang, dengan nilai F1-score yang baik untuk kedua kelas.

4.  **Prediksi Data Baru:**
    *   Model `nb` digunakan untuk memprediksi kelangsungan hidup penumpang baru. Contoh input data baru: `[Pclass, Sex (0=female, 1=male), Age, SibSp, Parch, Fare, Embarked (0=C, 1=Q, 2=S)]`.
    *   Model tidak hanya memberikan prediksi kelas (0 = Tidak Selamat, 1 = Selamat), tetapi juga probabilitas prediksi untuk setiap kelas (`nb.predict_proba`).
    *   **Contoh Prediksi:** Untuk penumpang baru `[3, 0, 25.0, 0, 0, 7.25, 2]`, model memprediksi **Selamat** (kelas 1) dengan probabilitas `[0.3268794, 0.6731206]` (32.69% tidak selamat, 67.31% selamat).

Model Naive Bayes menunjukkan akurasi yang solid dan merupakan pilihan yang baik untuk masalah klasifikasi ini, terutama sebagai model dasar yang efisien.
