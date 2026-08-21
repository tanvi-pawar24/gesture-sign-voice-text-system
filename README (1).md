# Gesture Sign-Based Voice and Text Conversion System

Assistive communication device that recognizes hand gestures in real time via camera and converts them into text (displayed on an LCD) and speech output. Built on Raspberry Pi B+ for speech- and hearing-impaired users.

## Features
- CNN-based hand gesture classifier (95% test accuracy)
- Real-time inference optimized with TensorFlow Lite (<1 second latency)
- OpenCV-based hand region segmentation using HSV skin-color thresholding
- Text-to-speech output via espeak/pyttsx3
- 16x2 I2C LCD display for recognized gesture text

## Tech Stack
Python, OpenCV, TensorFlow/Keras, TensorFlow Lite, Raspberry Pi B+, I2C LCD

## How It Works
```
[Hand Gesture] → [Pi Camera] → [OpenCV Preprocessing] → [CNN Classifier]
                                                                ↓
                                                    [Recognized Gesture Text]
                                                          ↓         ↓
                                                    [LCD Display] [Text-to-Speech]
                                                                     ↓
                                                                [Speaker]
```

1. Camera captures the hand gesture
2. OpenCV preprocesses the frame and isolates the hand region (skin-color HSV masking + contour detection)
3. A CNN model classifies the gesture in real time
4. Recognized gesture is displayed as text on the LCD
5. Text is converted to speech and played through the speaker

## Files
- `src/gesture_baseline.py` — initial version using Keras model and pyttsx3 for text-to-speech
- `src/pi_inference.py` — optimized version using TensorFlow Lite for faster on-device inference, with hand segmentation, confidence thresholding, and speak-cooldown logic to avoid repeated audio
- `docs/project_report.pdf` — full project report with hardware details, methodology, and results
- `requirements.txt` — Python dependencies

## Hardware Used
- Raspberry Pi B+
- Pi Camera Module
- 16x2 I2C LCD Display
- Speaker (3.5mm audio jack)
- 5V Power Adapter

## Results
Successfully recognized gestures: One, Two, Three, Fist, Palm
- 95% accuracy on test set
- Real-time detection latency under 1 second

## Applications
- Assistive communication device for speech/hearing-impaired individuals
- Smart home gesture control
- Human-Computer Interaction interfaces
- Gesture-based authentication systems

## Future Scope
- Expand gesture vocabulary to Indian/American Sign Language (ISL/ASL)
- Integrate CNN + LSTM for dynamic gesture sequences
- Add Bluetooth/WiFi connectivity
- Cloud-based voice generation

## Author
Tanvi Pawar
