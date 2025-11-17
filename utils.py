import cv2
import numpy as np

def crop_and_extract_hsv(image_bgr):
    """
    Preprocessing final untuk deteksi cabai rawit.
    Versi stabil dan toleran terhadap variasi warna serta pencahayaan.
    """

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Mask warna khas cabai (merah, oranye, hijau)
    lower_red1 = np.array([0, 40, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 30])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    lower_orange = np.array([11, 40, 30])
    upper_orange = np.array([30, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    lower_green = np.array([31, 35, 30])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Gabungkan semua mask
    mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_orange, mask_green))

    # Bersihkan noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Temukan kontur terbesar
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # Validasi area minimum
    h_img, w_img = image_bgr.shape[:2]
    if area < 0.005 * (h_img * w_img):  # 0.5% dari area gambar
        return None, False

    # Validasi bentuk (cabai umumnya agak memanjang, tapi fleksibel)
    x, y, w, h = cv2.boundingRect(cnt)
    ratio = max(w, h) / (min(w, h) + 1e-5)
    if ratio < 1.8:  # lebih toleran dari versi sebelumnya
        return None, False

    # Crop area cabai dengan padding
    pad = 8
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, image_bgr.shape[1] - x)
    h = min(h + 2 * pad, image_bgr.shape[0] - y)
    cropped = image_bgr[y:y + h, x:x + w]

    # Resize ke ukuran standar
    resized = cv2.resize(cropped, (150, 150))

    # Konversi ke HSV kembali
    hsv_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask_crop = cv2.inRange(hsv_crop, np.array([0, 30, 20]), np.array([180, 255, 255]))

    if np.count_nonzero(mask_crop) == 0:
        return None, False

    # Hitung rata-rata HSV hanya pada area valid
    h, s, v = cv2.split(hsv_crop)
    h_vals = h[mask_crop > 0]
    s_vals = s[mask_crop > 0]
    v_vals = v[mask_crop > 0]

    if h_vals.size == 0 or s_vals.size == 0 or v_vals.size == 0:
        return None, False

    h_mean = float(np.mean(h_vals))
    s_mean = float(np.mean(s_vals))
    v_mean = float(np.mean(v_vals))

    # Validasi akhir lebih fleksibel
    if np.isnan(h_mean) or s_mean < 15 or v_mean < 15:
        return None, False

    # Hue cabai: 0–95 (merah, oranye, hijau) → lebih luas agar merah gelap tidak ditolak
    if not (0 <= h_mean <= 95 or 150 <= h_mean <= 180):
        return None, False

    return (h_mean, s_mean, v_mean), True
