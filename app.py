from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from PIL import Image
import base64

from utils import crop_and_extract_hsv

app = Flask(__name__)

# 🔧 Load model
model = pickle.load(open("model/model_knn.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


# -------------------------------
# FIX WARNA UNTUK DISPLAY
# -------------------------------
def img_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# ----------------------------------------
# NORMALISASI CAHAYA (ANTI TERANG/GELAP)
# ----------------------------------------
def normalize_hsv(h, s, v):
    """
    Membuat HSV lebih stabil terhadap cahaya.
    """
    # Skala ulang V agar tidak terlalu gelap atau terlalu terang
    v_norm = np.interp(v, [0, 255], [60, 200])

    # Stabilkan S agar tidak drop pada cahaya tinggi
    s_norm = np.interp(s, [0, 255], [40, 255])

    return h, s_norm, v_norm


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    hasil_panen = None
    h_mean = s_mean = v_mean = None
    original64 = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            # Load gambar
            image = Image.open(file.stream)
            img_rgb = np.array(image.convert("RGB"))
            img_bgr = img_rgb[:, :, ::-1]

            # Tampilkan asli
            original64 = img_to_base64(img_rgb)

            # Ekstraksi HSV
            fitur, valid = crop_and_extract_hsv(img_bgr)

            if not valid:
                return render_template(
                    "index.html",
                    hasil="BUKAN CABAI",
                    hasil_panen="❌ Bukan objek cabai",
                    h=None, s=None, v=None,
                    original=original64
                )

            h_mean, s_mean, v_mean = fitur

            # -------------------------------
            # NORMALISASI HSV ANTI CAHAYA
            # -------------------------------
            h_fix, s_fix, v_fix = normalize_hsv(h_mean, s_mean, v_mean)

            fitur_fixed = np.array([h_fix, s_fix, v_fix])

            # Tolak gambar gelap ekstrem
            if v_fix < 30:
                return render_template(
                    "index.html",
                    hasil="BUKAN CABAI",
                    hasil_panen="❌ Terlalu gelap",
                    h=h_fix, s=s_fix, v=v_fix,
                    original=original64
                )

            # -----------------------------------
            # Prediksi KNN
            # -----------------------------------
            fitur_scaled = scaler.transform([fitur_fixed])
            pred = model.predict(fitur_scaled)
            label = label_encoder.inverse_transform(pred)[0]


            # ======================================================
            # ATURAN KOREKSI HSV (Versi Anti Cahaya)
            # ======================================================
            """
            Keterangan rentang tahan cahaya:
            - MATANG           : h 0–10 atau 160–180, s tinggi
            - SETENGAH MATANG  : h 11–22 jika s tinggi
            - MENTAH           : h 23–70
            """

            if (0 <= h_fix <= 10) or (160 <= h_fix <= 180):
                label = "matang"

            elif 11 <= h_fix <= 22:
                label = "setengah_matang"

            elif 23 <= h_fix <= 70:
                label = "mentah"


            hasil = label.upper()

            # ------------------------------
            # Status panen
            # ------------------------------
            if label == "matang":
                hasil_panen = "✅ <b>Layak Panen</b>"
            elif label == "setengah_matang":
                hasil_panen = "⚠️ <b>Bisa Dipanen</b> (setengah matang)"
            else:
                hasil_panen = "❌ <b>Belum Bisa Dipanen</b>"

    return render_template(
        "index.html",
        hasil=hasil,
        hasil_panen=hasil_panen,
        h=f"{h_mean:.2f}" if h_mean is not None else None,
        s=f"{s_mean:.2f}" if s_mean is not None else None,
        v=f"{v_mean:.2f}" if v_mean is not None else None,
        original=original64
    )


if __name__ == "__main__":
    app.run(debug=True)
