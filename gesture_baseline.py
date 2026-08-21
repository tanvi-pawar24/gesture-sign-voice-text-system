"""
gesture_baseline.py

Baseline gesture recognition script.
Captures hand gestures via camera, classifies them using a trained
Keras CNN model, displays the result on an I2C LCD, and speaks the
recognized gesture using pyttsx3.

Run on Raspberry Pi B+ with Pi Camera or USB webcam.
"""

import cv2
import numpy as np
import tensorflow.keras as keras
import pyttsx3
import I2C_LCD_driver
import time

# Load trained ML model
model = keras.models.load_model("gesture_model.h5")

# Initialize LCD
lcd = I2C_LCD_driver.lcd()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open Pi Camera / USB camera
cap = cv2.VideoCapture(0)

GESTURES = ["ONE", "TWO", "THREE", "FIST", "PALM"]

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    img = resized.reshape(1, 64, 64, 1) / 255.0

    # Prediction
    pred = np.argmax(model.predict(img))
    gesture_text = GESTURES[pred]

    # Display on LCD
    lcd.lcd_display_string("Gesture: ", 1)
    lcd.lcd_display_string(gesture_text, 2)

    # Speak
    engine.say(gesture_text)
    engine.runAndWait()

    # Show on screen for testing
    cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
