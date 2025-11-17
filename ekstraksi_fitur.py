import cv2
import numpy as np
import os
import pandas as pd

# Folder hasil crop
base_folder = "data_crop"
classes = ["matang", "mentah", "setengah_matang","non_cabai"]

data = []

for cls in classes:
    folder = os.path.join(base_folder, cls)
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            fitur = {
                "H_mean": np.mean(h),
                "S_mean": np.mean(s),
                "V_mean": np.mean(v),
                "H_std": np.std(h),
                "S_std": np.std(s),
                "V_std": np.std(v),
                "Label": cls
            }
            data.append(fitur)

# Simpan ke CSV
df = pd.DataFrame(data)
df.to_csv("data_fitur_cabai.csv", index=False)

print("✅ Ekstraksi fitur selesai dan disimpan sebagai data_fitur_cabai.csv")
