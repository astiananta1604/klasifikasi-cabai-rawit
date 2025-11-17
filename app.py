import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
from utils import crop_and_extract_hsv  # versi baru dengan validasi

# 🔧 Load model dan encoder
model = pickle.load(open("model/model_knn.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# 🧭 Konfigurasi halaman Streamlit
st.set_page_config(page_title="Klasifikasi Cabai Rawit", page_icon="🌶️", layout="wide")

st.title("🌶️ SISTEM KLASIFIKASI TINGKAT KEMATANGAN CABAI RAWIT")

st.write("""
Website ini dibuat untuk mengklasifikasikan tingkat kematangan cabai rawit
(mentah, setengah matang, matang) menggunakan algoritma *K-Nearest Neighbors (KNN)*.
""")

st.markdown("---")

# 📘 Penjelasan
st.header("🎯 Tujuan")
st.write("""
- Membantu petani atau pengguna menentukan tingkat kematangan cabai rawit.  
- Menunjukkan penerapan *machine learning* berbasis citra dalam bidang pertanian.  
- Menyediakan sistem sederhana dan mudah digunakan melalui website.  
""")

st.header("⚙️ Cara Kerja Aplikasi")
st.markdown("""
1. Unggah gambar cabai rawit pada menu *Unggah & Klasifikasi*.  
2. Sistem akan melakukan *preprocessing* seperti crop, resize, dan ekstraksi warna HSV.  
3. Nilai *Hue, Saturation, Value* dinormalisasi menggunakan *scaler*.  
4. Algoritma *K-Nearest Neighbor* (KNN) digunakan untuk menentukan kelas.  
5. Hasil ditampilkan: *Mentah*, *Setengah Matang*, atau *Matang*.  
""")

st.header("📤 Unggah & Klasifikasi Cabai Rawit")
st.write("""
Silakan **unggah gambar cabai rawit** yang ingin diklasifikasikan.  
Kamu dapat menarik dan melepaskan gambar ke kotak di bawah,  
atau klik area tersebut untuk memilih gambar dari perangkatmu.
""")

# 📸 Upload gambar
uploaded_file = st.file_uploader("🌶️ Tarik dan lepaskan atau pilih gambar cabai rawit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))[:, :, ::-1]  # Konversi PIL → OpenCV (BGR)

    st.image(image, caption="📸 Gambar yang diunggah", use_container_width=True)

    with st.spinner("🔍 Melakukan *processing* dan ekstraksi fitur..."):
        fitur, valid = crop_and_extract_hsv(img_array)

    if not valid:
        st.error("🚫 Gambar tidak valid. silahkan unggah gambar lain.")
    else:
        h_mean, s_mean, v_mean = fitur

        # Longgarkan validasi warna agar tidak menolak cabai merah gelap
        if s_mean < 15 or v_mean < 10:
            st.error("🚫 Gambar terlalu gelap atau saturasi terlalu rendah.")
        else:
            # 🔹 Normalisasi fitur
            fitur_scaled = scaler.transform([fitur])
            pred = model.predict(fitur_scaled)
            label = label_encoder.inverse_transform(pred)[0]

            # 🔹 Periksa inkonsistensi (opsional, bukan pengganti hasil)
            if (label == "mentah" and h_mean < 25):
                label = "setengah_matang"  # kemungkinan warna mulai berubah
            elif (label == "setengah_matang" and h_mean < 15):
                label = "matang"

            # ✂️ Tampilkan hasil crop
            hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 30, 0]), np.array([179, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                pad = 5
                x = max(x - pad, 0)
                y = max(y - pad, 0)
                w = min(w + 2 * pad, img_array.shape[1] - x)
                h = min(h + 2 * pad, img_array.shape[0] - y)
                cropped = img_array[y:y+h, x:x+w]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                st.image(cropped_rgb, caption="✂️ Gambar hasil crop", use_container_width=True)

            # 🧪 Tampilkan HSV & klasifikasi
            st.markdown(f"""
            ### 🧪 Rata-rata Nilai *Hue Saturation Value (HSV)*:
            - **Hue (H)**: `{h_mean:.2f}`
            - **Saturation (S)**: `{s_mean:.2f}`
            - **Value (V)**: `{v_mean:.2f}`
            """)

            # 📌 Tampilkan hasil klasifikasi
            label_lc = label.lower()
            if label_lc == "matang":
                panen_status = "✅ **Layak Panen** – cabai berwarna merah tua atau oranye pekat."
            elif label_lc == "setengah_matang":
                panen_status = "✅ **Layak Panen** – cabai mulai menunjukkan warna merah atau oranye."
            else:
                panen_status = "❌ **Belum Bisa Dipanen** – cabai masih berwarna hijau."

            st.success(f"📌 Hasil klasifikasi: **{label.upper()}**\n\n{panen_status}")
