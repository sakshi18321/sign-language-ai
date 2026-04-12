import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ── CONFIG ─────────────────────────────────────────────
DATASET_PATH = "dataset"   # ✅ YOUR FOLDER NAME
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ── INIT MEDIAPIPE ─────────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3  # lower for better detection
)

X = []
Y = []

total_images = 0
detected_images = 0

print("🚀 Starting dataset processing...\n")

# ── LOAD DATASET ───────────────────────────────────────
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"📂 Processing class: {label}")

    for img_name in tqdm(os.listdir(label_path)):
        total_images += 1

        img_path = os.path.join(label_path, img_name)

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize for consistency
            img = cv2.resize(img, (640, 480))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = mp_hands.process(img_rgb)

            if result.multi_hand_landmarks:
                detected_images += 1

                landmarks = result.multi_hand_landmarks[0]
                flat = []

                for lm in landmarks.landmark:
                    flat.extend([lm.x, lm.y, lm.z])

                # Normalize (VERY IMPORTANT)
                flat = np.array(flat)
                flat = flat - flat[0]  # make relative to wrist

                X.append(flat)
                Y.append(label)

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

# ── DATA CHECK ─────────────────────────────────────────
print("\n📊 DATA SUMMARY")
print("Total images:", total_images)
print("Hands detected:", detected_images)

if len(X) == 0:
    print("❌ ERROR: No valid data found!")
    print("👉 Fix dataset or improve lighting/images")
    exit()

# ── ENCODE LABELS ──────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(Y)
X = np.array(X)

print("\n✅ Dataset ready")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# ── MODEL ──────────────────────────────────────────────
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Training started...\n")

history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# ── SAVE MODEL ─────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, "asl_model.h5"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump({
        "labels": list(le.classes_),
        "feature_dim": 63,
        "accuracy": float(max(history.history['accuracy']))
    }, f)

print("\n🎉 TRAINING COMPLETE!")
print("Model saved in /models")