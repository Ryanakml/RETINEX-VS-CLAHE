# Perbandingan Metode Pemrosesan Gambar: CLAHE vs Retinex

![Perbandingan CLAHE vs Retinex](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*2duBDhghVu8CmB2o632JrA.png)

## Deskripsi

Aplikasi ini adalah alat interaktif berbasis Streamlit yang memungkinkan pengguna untuk membandingkan dua metode pemrosesan gambar populer: CLAHE (Contrast Limited Adaptive Histogram Equalization) dan Retinex. Aplikasi ini dirancang untuk membantu pengguna memahami perbedaan antara kedua metode tersebut melalui perbandingan visual dan analisis metrik kualitas gambar.

## Fitur Utama

- **Upload Gambar**: Pengguna dapat mengunggah gambar dalam format JPG, JPEG, atau PNG.
- **Pemrosesan Gambar**: Aplikasi secara otomatis memproses gambar menggunakan metode CLAHE dan Retinex.
- **Tampilan Perbandingan**: Menampilkan gambar asli, hasil CLAHE, dan hasil Retinex secara berdampingan.
- **Analisis Histogram**: Menampilkan histogram distribusi pixel untuk setiap gambar.
- **Metrik Kualitas**: Menghitung dan menampilkan nilai PSNR (Peak Signal-to-Noise Ratio) dan SSIM (Structural Similarity Index) untuk setiap metode.
- **Visualisasi Metrik**: Menampilkan grafik perbandingan metrik PSNR dan SSIM.
- **Penjelasan Metode**: Menyediakan penjelasan tentang konsep matematis di balik CLAHE dan Retinex.
- **Parameter Kustom**: Memungkinkan pengguna untuk menyesuaikan parameter CLAHE dan Retinex.

## Metode yang Diimplementasikan

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE adalah metode peningkatan kontras yang bekerja pada region kecil dalam gambar, yang disebut "tiles", bukan pada keseluruhan gambar. Metode ini efektif untuk meningkatkan detail lokal dan mengatasi masalah over-amplification pada metode histogram equalization global.

### Retinex

Retinex adalah teori persepsi warna yang menggabungkan "Retina" dan "Cortex", dikembangkan oleh Edwin Land. Metode ini memisahkan gambar menjadi komponen pencahayaan dan reflektansi. Aplikasi ini mendukung dua varian Retinex:
- **SSR (Single-Scale Retinex)**: Menggunakan satu skala Gaussian untuk estimasi pencahayaan.
- **MSR (Multi-Scale Retinex)**: Menggunakan beberapa skala Gaussian dan menggabungkan hasilnya.

## Persyaratan Sistem

- Python 3.6 atau lebih tinggi
- Streamlit 1.22.0
- OpenCV 4.7.0.72
- NumPy 1.24.3
- Matplotlib 3.7.1
- scikit-image 0.20.0
- Pillow 9.5.0
- pandas 2.0.1

## Instalasi

1. Clone repositori ini atau download file-file yang diperlukan.
2. Instal dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

## Cara Penggunaan

1. Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

2. Buka browser web dan akses URL yang ditampilkan (biasanya http://localhost:8501).
3. Upload gambar menggunakan panel di sidebar kiri.
4. Sesuaikan parameter untuk metode CLAHE dan Retinex sesuai kebutuhan.
5. Lihat hasil pemrosesan dan perbandingan metrik.
6. Baca penjelasan tentang metode dan interpretasi hasil.

## Struktur Aplikasi

Aplikasi ini terdiri dari beberapa komponen utama:

- **Fungsi Pemrosesan Gambar**:
  - `apply_clahe()`: Menerapkan metode CLAHE pada gambar
  - `ssr()`: Menerapkan Single-Scale Retinex
  - `msr()`: Menerapkan Multi-Scale Retinex

- **Fungsi Analisis**:
  - `plot_histogram()`: Menampilkan histogram distribusi pixel
  - `calculate_metrics()`: Menghitung metrik PSNR dan SSIM

- **Fungsi Tampilan**:
  - `display_method_explanation()`: Menampilkan penjelasan tentang metode
  - `display_method_comparison()`: Menampilkan perbandingan antara metode

- **Fungsi Utama**:
  - `main()`: Mengatur alur aplikasi dan antarmuka pengguna

## Perbandingan Metode

| Aspek | CLAHE | Retinex |
|-------|-------|---------|
| **Prinsip Dasar** | Meningkatkan kontras lokal dengan histogram equalization adaptif | Memisahkan komponen pencahayaan dan reflektansi |
| **Kelebihan** | Meningkatkan detail lokal, mengatasi over-amplification | Menangani pencahayaan tidak merata, mempertahankan warna global |
| **Kelemahan** | Dapat meningkatkan noise, kurang efektif pada pencahayaan tidak merata | Komputasi lebih berat, sensitif terhadap parameter |
| **Aplikasi Ideal** | Gambar medis, detail tekstur | Fotografi dengan pencahayaan buruk, gambar dengan bayangan |
| **Kompleksitas** | Sedang | Tinggi |

## Interpretasi Metrik

- **PSNR (Peak Signal-to-Noise Ratio)**: Mengukur kualitas rekonstruksi gambar. Nilai lebih tinggi menunjukkan kemiripan yang lebih baik dengan gambar asli.
- **SSIM (Structural Similarity Index)**: Mengukur kemiripan struktural antara dua gambar. Nilai mendekati 1 menunjukkan kemiripan yang lebih tinggi.

## Kontribusi

Kontribusi untuk meningkatkan aplikasi ini sangat diterima. Silakan fork repositori ini, buat perubahan, dan kirimkan pull request.

---

Dibuat oleh saya sendiri menggunakan Streamlit dan Python.

        
