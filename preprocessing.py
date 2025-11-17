import os
import cv2
import numpy as np
import pandas as pd

# Path ke folder dataset
dataset_path = 'data_asli'
categories = ['mentah', 'setengah_matang', 'matang']
data = []

# Loop setiap folder kategori
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = category  # nama kelas sebagai label
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        # Baca gambar
        img = cv2.imread(img_path)
        if img is None:
            continue  # skip jika gambar tidak terbaca

        # Resize agar seragam
        img = cv2.resize(img, (150, 150))

        # Konversi ke HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Ambil rata-rata tiap channel
        H_mean = np.mean(hsv_img[:, :, 0])
        S_mean = np.mean(hsv_img[:, :, 1])
        V_mean = np.mean(hsv_img[:, :, 2])

        # Simpan fitur dan label
        data.append([H_mean, S_mean, V_mean, label])

# Simpan hasil ekstraksi ke CSV
df = pd.DataFrame(data, columns=['H_mean', 'S_mean', 'V_mean', 'label'])
df.to_csv('data_fitur_cabai.csv', index=False)

print("✅ Preprocessing selesai. File data_fitur_cabai.csv berhasil dibuat dengan fitur HSV.")
