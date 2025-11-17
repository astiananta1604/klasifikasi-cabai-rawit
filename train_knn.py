import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import crop_and_extract_hsv

# === Konfigurasi dataset ===
dataset_path = 'data_asli'  # folder utama dataset
categories = ['mentah', 'setengah_matang', 'matang', 'non_cabai']

data = []
labels = []

print("🔍 Mulai ekstraksi fitur HSV dari gambar...")

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Gagal membaca gambar: {img_path}")
            continue

        # --- Preprocessing & ekstraksi fitur HSV ---
        hsv_features, valid = crop_and_extract_hsv(img)

        if valid:
            data.append(hsv_features)
            labels.append(category)
        else:
            print(f"❌ Gambar dilewati (bukan cabai valid): {filename}")

# Konversi ke DataFrame
df = pd.DataFrame(data, columns=['H_mean', 'S_mean', 'V_mean'])
df['Label'] = labels

print(f"\n✅ Total data valid: {len(df)}")
print(df.head())

# === Encoding label ===
label_encoder = LabelEncoder()
df['Label_encoded'] = label_encoder.fit_transform(df['Label'])

X = df[['H_mean', 'S_mean', 'V_mean']].values
y = df['Label_encoded'].values

# === Standarisasi fitur ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Pembagian data train-test ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# === Pelatihan model KNN ===
k = 5  # nilai K bisa disesuaikan
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# === Evaluasi ===
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("\n📊 Model KNN berhasil dilatih!")
print(f"Akurasi: {acc:.2f}\n")
print("Confusion Matrix:")
print(cm)
print("\nLaporan Klasifikasi:")
print(report)

# === Simpan model ===
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/model_knn.pkl', 'wb'))
pickle.dump(label_encoder, open('model/label_encoder.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

print("\n💾 Model, scaler, dan label encoder berhasil disimpan ke folder 'model'!")
