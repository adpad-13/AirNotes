# AirNotes ✍️

> Write in the air. Read it as text.

AirNotes is a real-time AR application that tracks hand gestures via 
webcam to let users write characters in mid-air, converting them into 
digital text using computer vision and deep learning.

No stylus. No touchscreen. Just your hand.

---

## How It Works

**M1 — Hand Tracking**  
MediaPipe Hand Landmarker detects 21 landmarks on the hand in real time.
Pen state is determined by the Euclidean distance between the thumb tip 
(landmark 4) and index fingertip (landmark 8) — pinch to write, 
release to lift.

**M2 — Stroke Processing**  
Strokes are rendered on a canvas overlay using a 3-frame sliding window 
moving average to eliminate coordinate jitter. Hysteresis thresholding 
(0.025 on / 0.045 off) prevents pen state flickering. Characters are 
automatically extracted after 1.5s of pen inactivity — cropped to the 
bounding box, converted to grayscale, and resized to 28×28px.

**M3 — Character Recognition** *(in progress)*  
CNN trained on the EMNIST dataset 800 thousand+ images for 
real-time classification of air-written characters. Output assembles 
into words and sentences, exported as .txt or .pdf.

---

## Currently implemented Features
- Real-time hand tracking MediaPipe
- Pinch gesture pen detection with hysteresis threshold
- Smooth stroke rendering — 3-frame moving average
- Automatic character segmentation and extraction
- 28×28 grayscale output ready for CNN inference
- AR overlay — strokes drawn directly over live camera feed
- Auto-canvas clear on pen lift (1.5s timeout)

---

## Planned Features
- CNN character recognition (EMNIST → fine-tuned on custom data)
- Word-level segmentation and text assembly
- PDF / .txt export pipeline
- Autocorrect and spell-check layer
- Voice recognition integration
- Emoji identification from drawn symbols
- Mobile and AR headset support

---

## Tech Stack
| Component | Technology |
|---|---|
| Hand Tracking | MediaPipe Hand Landmarker |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Numerical Processing | NumPy |
| Language | Python 3.10+ |

---

## Setup

```bash
git clone https://github.com/adpad-13/AirNotes.git
cd AirNotes
pip install mediapipe opencv-python torch numpy
```

Download the MediaPipe hand landmarker model:
```bash
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```



## Usage
- Point your hand at the webcam
- Pinch thumb and index finger together to start writing
- Release pinch to lift pen
- Wait 1.5 seconds after finishing a character for extraction
- Press `q` to quit

---

## Project Status
**M1 Complete** — Hand tracking, pen detection, smooth drawing, 
character extraction pipeline working at native camera fps  
**M2 In Progress** — CNN training on EMNIST  
**M3 Planned** — Text assembly, word segmentation, export pipeline

---

## Author
**Aditya Singh** — 3rd year CS student at VIT AP  
Building at the intersection of computer vision and NLP  
