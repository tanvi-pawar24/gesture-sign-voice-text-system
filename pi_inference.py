"""
pi_inference.py

Optimized gesture recognition script for real-time deployment on
Raspberry Pi. Uses a TensorFlow Lite model for faster on-device
inference, OpenCV HSV skin-color thresholding for hand region
segmentation, and confidence-threshold + cooldown logic to avoid
repeated speech output.
"""

import cv2
import numpy as np
import time
import subprocess
from tflite_runtime.interpreter import Interpreter
from lcd_i2c import I2CLcd

# ------------ User params ------------
TFLITE_MODEL = "gesture_model.tflite"
LABELS = "labels.txt"
USE_PICAMERA = False  # If using /dev/video0 USB webcam set False.
CAM_INDEX = 0
INPUT_SIZE = (224, 224)
SPEAK_COOLDOWN = 2.0  # seconds between spoken outputs
THRESHOLD = 0.6
# -------------------------------------

# Load labels
with open(LABELS, "r") as f:
    labels = [l.strip() for l in f.readlines()]

# Load TFLite model
interpreter = Interpreter(TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# If model expects uint8, track scale/zero_point
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']

# Create LCD instance
lcd = I2CLcd()
lcd.clear()
lcd.write_line("Gesture Ready", 1)
time.sleep(0.5)
lcd.clear()

# Video capture
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_spoken_time = 0
last_label = None


def preprocess(frame):
    # frame is BGR from cv2
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    if input_dtype == np.uint8:
        img = (img).astype(np.uint8)
    else:
        img = img.astype(np.float32) / 255.0
    return img


def speak(text):
    # use espeak (offline)
    subprocess.Popen(['espeak', f"\"{text}\""])


def run_inference(img):
    inp = preprocess(img)
    # shape: [1,h,w,3]
    inp = np.expand_dims(inp, axis=0)
    if input_dtype == np.uint8:
        inp = inp.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])

    # handle float or uint8 output by dequantizing if needed
    if output_dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        out = scale * (out - zero_point)

    probs = out[0]
    return probs


def get_hand_roi(frame):
    # simple skin-color thresholding in HSV as a quick ROI
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours and largest bbox
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            # pad
            pad = 20
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            roi = frame[y0:y1, x0:x1]
            return roi, (x0, y0, x1, y1)

    return None, None


print("Starting inference. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera")
        break

    roi, bbox = get_hand_roi(frame)
    display_text = "No hand"
    conf = 0.0

    if roi is not None:
        probs = run_inference(roi)
        idx = np.argmax(probs)
        conf = float(probs[idx])
        display_text = f"{labels[idx]} ({conf:.2f})"

        # speak if high confidence and changed label and cooldown passed
        now = time.time()
        if conf >= THRESHOLD and (labels[idx] != last_label or (now - last_spoken_time) > SPEAK_COOLDOWN):
            lcd.clear()
            lcd.write_line(labels[idx], 1)
            lcd.write_line(f"{conf:.2f}", 2)
            speak(labels[idx])
            last_spoken_time = now
            last_label = labels[idx]
    else:
        lcd.clear()
        lcd.write_line("Show Hand", 1)

    # Draw bbox and text on frame for debugging and display
    if bbox:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
