import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp

# Load trained model
model = tf.keras.models.load_model("model.h5")

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
def detect_sign(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return "No Hand", 0.0

    # Get bounding box of hand
    h, w, _ = image.shape
    landmarks = results.multi_hand_landmarks[0]

    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Crop hand only
    hand_img = image[y_min:y_max, x_min:x_max]

    if hand_img.size == 0:
        return "No Hand", 0.0

    # Resize and predict
    hand_img = cv2.resize(hand_img, (64, 64))
    hand_img = hand_img / 255.0
    hand_img = np.expand_dims(hand_img, axis=0)

    pred = model.predict(hand_img, verbose=0)
    index = np.argmax(pred)
    confidence = float(np.max(pred))

    return classes[index], confidence