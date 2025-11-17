import cv2
import os
import numpy as np

# === Konfigurasi utama ===
folder_asal = 'data_asli'      # folder berisi mentah, setengah_matang, matang, non_cabai
folder_tujuan = 'data_resize'  # folder hasil resize dan enhancement
ukuran = (150, 150)            # ukuran gambar untuk model KNN

# === Buat folder tujuan jika belum ada ===
os.makedirs(folder_tujuan, exist_ok=True)

# === Enhancement warna ===
def enhance_warna(img):
    """Meningkatkan kecerahan dan kejenuhan warna agar tampak seperti cabai asli"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # tingkatkan saturasi dan brightness dengan hati-hati
    s = cv2.add(s, 25)  # tambah kejenuhan warna
    v = cv2.add(v, 20)  # tambah kecerahan

    hsv_enhanced = cv2.merge((h, s, v))
    img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return img_enhanced

# === Proses setiap kelas ===
for kelas in os.listdir(folder_asal):
    path_kelas_asal = os.path.join(folder_asal, kelas)
    path_kelas_tujuan = os.path.join(folder_tujuan, kelas)

    if os.path.isdir(path_kelas_asal):
        os.makedirs(path_kelas_tujuan, exist_ok=True)
        jumlah_gambar = 0

        for nama_file in os.listdir(path_kelas_asal):
            if nama_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path_asal = os.path.join(path_kelas_asal, nama_file)
                path_tujuan = os.path.join(path_kelas_tujuan, nama_file)

                img = cv2.imread(path_asal)
                if img is not None:
                    # ubah ukuran
                    img_resize = cv2.resize(img, ukuran)

                    # enhancement warna agar terlihat seperti cabai aslinya
                    img_final = enhance_warna(img_resize)

                    # simpan hasil
                    cv2.imwrite(path_tujuan, img_final)
                    jumlah_gambar += 1
                    print(f"✅ {kelas}/{nama_file} berhasil diproses")
                else:
                    print(f"❌ Gagal membaca gambar: {kelas}/{nama_file}")

        print(f"📸 Total gambar pada kelas '{kelas}': {jumlah_gambar}\n")

print("🎯 Proses resize dan enhancement selesai!")
