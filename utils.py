import cv2
import numpy as np

def crop_and_extract_hsv(image_bgr):
    """
    Preprocessing final untuk klasifikasi cabai rawit.
    Versi ini sangat stabil untuk:
    - mentah (hijau – kuning hijau)
    - setengah matang (oranye)
    - matang (merah)
    - non-cabai → INVALID
    """

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # ================================
    # MASK WARNA CABAI — versi terbaik
    # ================================
    # Merah
    lower_red1 = np.array([0, 40, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 30])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # Oranye (disempitkan!)
    lower_orange = np.array([15, 45, 35])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Hijau (diperluas untuk mengamankan mentah)
    lower_green = np.array([26, 35, 30])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Gabungkan
    mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_orange, mask_green))

    # ================================
    # NOISE CLEANING
    # ================================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ================================
    # CARI KONTOUR
    # ================================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    h_img, w_img = image_bgr.shape[:2]
    if area < 0.005 * (h_img * w_img):
        return None, False

    # ================================
    # VALIDASI BENTUK CABAI (memanjang)
    # ================================
    x, y, w, h = cv2.boundingRect(cnt)
    ratio = max(w, h) / (min(w, h) + 1e-5)

    if ratio < 1.8:  # tetap longgar namun aman
        return None, False

    # ================================
    # CROP
    # ================================
    pad = 8
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, image_bgr.shape[1] - x)
    h = min(h + 2 * pad, image_bgr.shape[0] - y)

    cropped = image_bgr[y:y + h, x:x + w]
    resized = cv2.resize(cropped, (150, 150))

    # ================================
    # HITUNG HSV
    # ================================
    hsv_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Mask agar hanya area cabai yg dihitung
    mask_crop = cv2.inRange(hsv_crop, np.array([0, 30, 20]), np.array([180, 255, 255]))
    if np.count_nonzero(mask_crop) == 0:
        return None, False

    h, s, v = cv2.split(hsv_crop)
    h_vals = h[mask_crop > 0]
    s_vals = s[mask_crop > 0]
    v_vals = v[mask_crop > 0]

    if h_vals.size == 0:
        return None, False

    h_mean = float(np.median(h_vals))
    s_mean = float(np.median(s_vals))
    v_mean = float(np.median(v_vals))

    # Validasi minimal cahaya
    if v_mean < 20 or s_mean < 15:
        return None, False

    # ================================
    # VALIDASI HUE KHUSUS CABAI
    # ================================
    # Merah
    if (0 <= h_mean <= 7) or (160 <= h_mean <= 180):
        return (h_mean, s_mean, v_mean), True  # MATANG

    # Setengah matang (oranye)
    if 12 <= h_mean <= 24 and s_mean >= 60:
        return (h_mean, s_mean, v_mean), True  # SETENGAH_MATANG

    # Mentah (hijau – kuning hijau)
    if 25 <= h_mean <= 75 and s_mean >= 50:
        return (h_mean, s_mean, v_mean), True  # MENTAH

    # Selain hue cabai → INVALID
    return None, False
