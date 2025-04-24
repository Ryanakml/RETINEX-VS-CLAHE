import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io
import base64

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Perbandingan CLAHE vs Retinex",
    page_icon="🖼️",
    layout="wide"
)

# Judul aplikasi
st.title("Perbandingan Metode Pemrosesan Gambar: CLAHE vs Retinex")
st.markdown("Aplikasi ini memungkinkan Anda membandingkan dua metode pemrosesan gambar populer untuk meningkatkan kualitas gambar.")

# Fungsi untuk menerapkan CLAHE
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Konversi ke LAB untuk memisahkan channel luminance
    if len(image.shape) == 3:  # Gambar berwarna
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Terapkan CLAHE pada channel L
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Gabungkan kembali channel
        limg = cv2.merge((cl, a, b))
        
        # Konversi kembali ke RGB
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:  # Gambar grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(image)
    
    return enhanced_img

# Fungsi untuk menerapkan Single-Scale Retinex (SSR)
def ssr(image, sigma=80):
    if len(image.shape) == 3:  # Gambar berwarna
        result = np.zeros_like(image, dtype=np.float64)
        for i in range(3):
            channel = image[:, :, i].astype(np.float64)
            # Tambahkan epsilon untuk menghindari log(0)
            channel += 1.0
            # Terapkan Gaussian blur
            blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
            # Rumus Retinex: R(x,y) = log(I(x,y)) - log(F(x,y)*I(x,y))
            result[:, :, i] = np.log10(channel) - np.log10(blurred)
        
        # Normalisasi hasil
        result = (result - result.min()) / (result.max() - result.min()) * 255
        result = result.astype(np.uint8)
    else:  # Gambar grayscale
        image_float = image.astype(np.float64) + 1.0
        blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)
        result = np.log10(image_float) - np.log10(blurred)
        result = (result - result.min()) / (result.max() - result.min()) * 255
        result = result.astype(np.uint8)
    
    return result

# Fungsi untuk menerapkan Multi-Scale Retinex (MSR)
def msr(image, sigmas=[15, 80, 250]):
    if len(image.shape) == 3:  # Gambar berwarna
        result = np.zeros_like(image, dtype=np.float64)
        for i in range(3):
            channel = image[:, :, i].astype(np.float64)
            channel += 1.0
            
            retinex = np.zeros_like(channel)
            for sigma in sigmas:
                blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
                retinex += np.log10(channel) - np.log10(blurred)
            
            # Rata-rata hasil dari semua skala
            retinex = retinex / len(sigmas)
            result[:, :, i] = retinex
        
        # Normalisasi hasil
        result = (result - result.min()) / (result.max() - result.min()) * 255
        result = result.astype(np.uint8)
    else:  # Gambar grayscale
        image_float = image.astype(np.float64) + 1.0
        retinex = np.zeros_like(image_float)
        
        for sigma in sigmas:
            blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)
            retinex += np.log10(image_float) - np.log10(blurred)
        
        retinex = retinex / len(sigmas)
        result = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
        result = result.astype(np.uint8)
    
    return result

# Fungsi untuk menghitung dan menampilkan histogram
def plot_histogram(image, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    
    if len(image.shape) == 3:  # Gambar berwarna
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    else:  # Gambar grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray')
    
    ax.set_xlim([0, 256])
    ax.set_title(title)
    ax.set_xlabel('Nilai Pixel')
    ax.set_ylabel('Frekuensi')
    
    return fig

# Fungsi untuk menghitung metrik kualitas gambar
def calculate_metrics(original, processed):
    # Konversi ke grayscale jika berwarna untuk perhitungan SSIM
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # Hitung PSNR
    psnr_value = psnr(original, processed)
    
    # Hitung SSIM
    ssim_value = ssim(original_gray, processed_gray, data_range=processed_gray.max() - processed_gray.min())
    
    return psnr_value, ssim_value

# Fungsi untuk menampilkan gambar dengan caption
def display_image_with_caption(image, caption):
    st.image(image, caption=caption, use_column_width=True)

# Fungsi untuk menampilkan penjelasan metode
def display_method_explanation(method):
    if method == "CLAHE":
        st.markdown("""
        ### Penjelasan Metode CLAHE
        
        **CLAHE (Contrast Limited Adaptive Histogram Equalization)** adalah metode peningkatan kontras yang bekerja pada region kecil dalam gambar, yang disebut "tiles", bukan pada keseluruhan gambar.
        
        **Konsep Matematis:**
        1. Gambar dibagi menjadi blok-blok kecil (tiles)
        2. Histogram equalization diterapkan pada setiap blok
        3. Untuk mencegah noise berlebih, histogram dibatasi (clipping)
        4. Bilinear interpolation diterapkan untuk menghilangkan batas artifisial antar blok
        
        **Formula:**
        - Untuk setiap tile, histogram dihitung dan dibatasi pada clip limit
        - Kelebihan pixel didistribusikan secara merata ke seluruh histogram
        - CDF (Cumulative Distribution Function) dihitung untuk pemetaan nilai pixel
        
        CLAHE sangat efektif untuk meningkatkan detail lokal dan mengatasi masalah over-amplification pada metode histogram equalization global.
        """)
    elif method == "Retinex":
        st.markdown("""
        ### Penjelasan Metode Retinex
        
        **Retinex** adalah teori persepsi warna yang menggabungkan "Retina" dan "Cortex", dikembangkan oleh Edwin Land. Metode ini memisahkan gambar menjadi komponen pencahayaan dan reflektansi.
        
        **Konsep Matematis:**
        1. Gambar input I(x,y) dianggap sebagai produk dari pencahayaan L(x,y) dan reflektansi R(x,y):
           I(x,y) = L(x,y) × R(x,y)
        2. Dengan mengambil logaritma:
           log(I) = log(L) + log(R)
        3. Pencahayaan L diestimasi dengan Gaussian blur pada gambar input
        4. Reflektansi R dihitung dengan:
           log(R) = log(I) - log(L)
        
        **Varian Retinex:**
        - **SSR (Single-Scale Retinex)**: Menggunakan satu skala Gaussian untuk estimasi pencahayaan
        - **MSR (Multi-Scale Retinex)**: Menggunakan beberapa skala Gaussian dan menggabungkan hasilnya
        
        Retinex sangat baik dalam meningkatkan detail pada area gelap sambil mempertahankan warna global dan mengurangi efek pencahayaan yang tidak merata.
        """)

# Fungsi untuk menampilkan perbandingan metode
def display_method_comparison():
    st.markdown("""
    ### Perbandingan CLAHE vs Retinex
    
    | Aspek | CLAHE | Retinex |
    |-------|-------|---------|
    | **Prinsip Dasar** | Meningkatkan kontras lokal dengan histogram equalization adaptif | Memisahkan komponen pencahayaan dan reflektansi |
    | **Kelebihan** | Meningkatkan detail lokal, mengatasi over-amplification | Menangani pencahayaan tidak merata, mempertahankan warna global |
    | **Kelemahan** | Dapat meningkatkan noise, kurang efektif pada pencahayaan tidak merata | Komputasi lebih berat, sensitif terhadap parameter |
    | **Aplikasi Ideal** | Gambar medis, detail tekstur | Fotografi dengan pencahayaan buruk, gambar dengan bayangan |
    | **Kompleksitas** | Sedang | Tinggi |
    
    **Interpretasi Metrik:**
    - **PSNR (Peak Signal-to-Noise Ratio)**: Mengukur kualitas rekonstruksi gambar. Nilai lebih tinggi menunjukkan kemiripan yang lebih baik dengan gambar asli.
    - **SSIM (Structural Similarity Index)**: Mengukur kemiripan struktural antara dua gambar. Nilai mendekati 1 menunjukkan kemiripan yang lebih tinggi.
    
    Kedua metode memiliki kekuatan masing-masing tergantung pada karakteristik gambar input dan tujuan pemrosesan.
    """)

# Fungsi utama aplikasi
def main():
    # Sidebar untuk upload gambar dan pengaturan
    st.sidebar.header("Upload Gambar")
    uploaded_file = st.sidebar.file_uploader("Pilih gambar untuk diproses", type=["jpg", "jpeg", "png"])
    
    # Parameter untuk metode pemrosesan
    st.sidebar.header("Parameter CLAHE")
    clahe_clip_limit = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1)
    clahe_tile_size = st.sidebar.slider("Tile Size", 2, 16, 8, 1)
    
    st.sidebar.header("Parameter Retinex")
    retinex_type = st.sidebar.radio("Tipe Retinex", ["Single-Scale (SSR)", "Multi-Scale (MSR)"])
    if retinex_type == "Single-Scale (SSR)":
        retinex_sigma = st.sidebar.slider("Sigma", 10, 300, 80, 5)
    else:
        st.sidebar.markdown("Menggunakan sigma default: 15, 80, 250")
    
    # Tampilkan penjelasan metode
    st.sidebar.header("Informasi Metode")
    method_info = st.sidebar.radio("Tampilkan penjelasan untuk:", ["CLAHE", "Retinex", "Perbandingan"])
    
    if method_info == "CLAHE":
        display_method_explanation("CLAHE")
    elif method_info == "Retinex":
        display_method_explanation("Retinex")
    else:
        display_method_comparison()
    
    # Proses gambar jika ada yang diupload
    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Konversi ke RGB jika RGBA
        if image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Terapkan metode pemrosesan
        clahe_result = apply_clahe(image_np, clahe_clip_limit, (clahe_tile_size, clahe_tile_size))
        
        if retinex_type == "Single-Scale (SSR)":
            retinex_result = ssr(image_np, retinex_sigma)
        else:
            retinex_result = msr(image_np)
        
        # Hitung metrik kualitas
        psnr_clahe, ssim_clahe = calculate_metrics(image_np, clahe_result)
        psnr_retinex, ssim_retinex = calculate_metrics(image_np, retinex_result)
        
        # Tampilkan gambar dan hasil
        st.header("Hasil Pemrosesan Gambar")
        
        # Buat tiga kolom untuk menampilkan gambar
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Gambar Asli")
            st.image(image_np, use_column_width=True)
            st.pyplot(plot_histogram(image_np, "Histogram Gambar Asli"))
        
        with col2:
            st.subheader("Hasil CLAHE")
            st.image(clahe_result, use_column_width=True)
            st.pyplot(plot_histogram(clahe_result, "Histogram CLAHE"))
            st.metric("PSNR", f"{psnr_clahe:.2f} dB")
            st.metric("SSIM", f"{ssim_clahe:.4f}")
        
        with col3:
            st.subheader("Hasil Retinex")
            st.image(retinex_result, use_column_width=True)
            st.pyplot(plot_histogram(retinex_result, "Histogram Retinex"))
            st.metric("PSNR", f"{psnr_retinex:.2f} dB")
            st.metric("SSIM", f"{ssim_retinex:.4f}")
        
        # Tampilkan perbandingan metrik
        st.header("Perbandingan Metrik Kualitas")
        
        # Buat dataframe untuk perbandingan
        import pandas as pd
        metrics_df = pd.DataFrame({
            'Metode': ['CLAHE', 'Retinex'],
            'PSNR (dB)': [f"{psnr_clahe:.2f}", f"{psnr_retinex:.2f}"],
            'SSIM': [f"{ssim_clahe:.4f}", f"{ssim_retinex:.4f}"]
        })
        
        st.table(metrics_df)
        
        # Visualisasi perbandingan metrik
        st.subheader("Visualisasi Perbandingan Metrik")
        
        # Plot perbandingan PSNR
        fig_psnr, ax_psnr = plt.subplots(figsize=(8, 4))
        ax_psnr.bar(['CLAHE', 'Retinex'], [psnr_clahe, psnr_retinex])
        ax_psnr.set_ylabel('PSNR (dB)')
        ax_psnr.set_title('Perbandingan PSNR')
        st.pyplot(fig_psnr)
        
        # Plot perbandingan SSIM
        fig_ssim, ax_ssim = plt.subplots(figsize=(8, 4))
        ax_ssim.bar(['CLAHE', 'Retinex'], [ssim_clahe, ssim_retinex])
        ax_ssim.set_ylabel('SSIM')
        ax_ssim.set_title('Perbandingan SSIM')
        st.pyplot(fig_ssim)
        
        # Tambahkan penjelasan hasil
        st.header("Analisis Hasil")
        st.markdown(f"""
        ### Interpretasi Hasil
        
        Berdasarkan metrik yang dihitung:
        
        - **PSNR**: {'CLAHE' if psnr_clahe > psnr_retinex else 'Retinex'} menghasilkan nilai PSNR yang lebih tinggi ({max(psnr_clahe, psnr_retinex):.2f} dB), yang menunjukkan bahwa metode ini menghasilkan gambar yang lebih mirip dengan gambar asli dalam hal nilai pixel.
        
        - **SSIM**: {'CLAHE' if ssim_clahe > ssim_retinex else 'Retinex'} menghasilkan nilai SSIM yang lebih tinggi ({max(ssim_clahe, ssim_retinex):.4f}), yang menunjukkan bahwa metode ini lebih baik dalam mempertahankan struktur gambar asli.
        
        ### Kesimpulan
        
        - **CLAHE** cenderung meningkatkan kontras lokal dan detail, tetapi mungkin mengubah distribusi intensitas global.
        
        - **Retinex** lebih efektif dalam menangani pencahayaan yang tidak merata dan mempertahankan warna, tetapi mungkin menghasilkan gambar yang lebih berbeda dari aslinya.
        
        Pilihan metode terbaik bergantung pada karakteristik gambar input dan tujuan pemrosesan yang diinginkan.
        """)
    else:
        # Tampilkan instruksi jika belum ada gambar yang diupload
        st.info("Silakan upload gambar untuk memulai pemrosesan.")
        st.markdown("""
        ### Cara Menggunakan Aplikasi Ini
        
        1. Upload gambar menggunakan panel di sidebar kiri
        2. Sesuaikan parameter untuk metode CLAHE dan Retinex
        3. Lihat hasil pemrosesan dan perbandingan metrik
        4. Baca penjelasan tentang metode dan interpretasi hasil
        
        Aplikasi ini memungkinkan Anda membandingkan dua metode pemrosesan gambar populer:
        - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
        - **Retinex (Single-Scale atau Multi-Scale)**
        
        Kedua metode ini memiliki pendekatan yang berbeda untuk meningkatkan kualitas gambar, dan aplikasi ini membantu Anda memahami kelebihan dan kekurangan masing-masing.
        """)

if __name__ == "__main__":
    main()