import cv2
import numpy as np
import os
import shutil

# === Pengaturan folder input dan output ===
input_base = "data_asli"     # Folder berisi gambar mentah, setengah_matang, matang
output_base = "data_crop"    # Folder hasil crop
classes = ["mentah", "setengah_matang", "matang"]
resize_size = (150, 150)

# === Hapus folder lama hasil crop ===
if os.path.exists(output_base):
    shutil.rmtree(output_base)
os.makedirs(output_base, exist_ok=True)

# === Fungsi bantu untuk cropping cabai ===
def crop_chili(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Gabungan deteksi semua kemungkinan warna cabai (merah, oranye, hijau)
    lower_red1 = np.array([0, 100, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 60])
    upper_red2 = np.array([179, 255, 255])
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_orange = np.array([10, 80, 60])
    upper_orange = np.array([25, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Gabungkan semua warna potensial cabai
    mask = cv2.bitwise_or(mask_red, mask_green)
    mask = cv2.bitwise_or(mask, mask_orange)

    # Hilangkan noise kecil (morphological filtering)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Ambil kontur terbesar
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Tambahkan padding agar crop tidak terlalu ketat
    pad = int(0.1 * max(w, h))
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, img.shape[1] - x)
    h = min(h + 2 * pad, img.shape[0] - y)

    cropped = img[y:y+h, x:x+w]

    # Jika area terlalu kecil, anggap gagal
    if w < 40 or h < 40:
        return None
    return cropped


# === Proses setiap kelas ===
for cls in classes:
    input_path = os.path.join(input_base, cls)
    output_path = os.path.join(output_base, cls)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            cropped = crop_chili(img)

            # Jika gagal deteksi, fallback crop tengah
            if cropped is None:
                print(f"⚠️ Gagal deteksi objek cabai, fallback tengah: {filename}")
                h, w = img.shape[:2]
                cropped = img[h//4:3*h//4, w//4:3*w//4]

            # Resize proporsional tanpa distorsi
            h_c, w_c = cropped.shape[:2]
            scale = min(resize_size[0]/w_c, resize_size[1]/h_c)
            new_w, new_h = int(w_c * scale), int(h_c * scale)
            resized = cv2.resize(cropped, (new_w, new_h))

            # Tambahkan padding agar hasil final 150x150 dan latar putih
            top = (resize_size[1] - new_h) // 2
            bottom = resize_size[1] - new_h - top
            left = (resize_size[0] - new_w) // 2
            right = resize_size[0] - new_w - left

            final_img = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

            save_path = os.path.join(output_path, filename)
            cv2.imwrite(save_path, final_img)

print("✅ Semua gambar berhasil di-crop dan disimpan di folder 'data_crop'")
