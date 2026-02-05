

# 🚀  OCR System 

## 📌 Overview

This project is a **fully custom-built Offline Optical Character Recognition (OCR) system**, developed **from scratch** without using existing OCR engines (e.g., PaddleOCR).
It is designed for **industrial inspection and validation use cases**, where **accuracy, compliance, and offline execution** are critical.

The system performs:

* Text **detection**
* Text **recognition** using a custom CRNN + CTC pipeline
* **Bounding box visualization**
* **Strict field-level validation (OK / NG)**
* Fully **offline deployment**

---

## 🎯 Key Objectives

* ✅ High accuracy with punctuation preservation
* ✅ Offline & edge-device compatible (Jetson / x86)
---

## 🧠 System Architecture

The OCR pipeline is divided into **three core stages**:

1. **Text Detection**

   * Detects text regions from input images
   * Outputs bounding boxes for each text line/word

2. **Text Recognition**

   * Custom CRNN (CNN + RNN) model
   * CTC loss for sequence prediction
   * Character-level decoding

3. **Validation & Visualization**

   * Field-wise rule checking (regex / length / format)
   * OK / NG status overlay on image
   * Bounding box + recognized text visualization

---



## ⚙️ Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Image Processing:** OpenCV, NumPy
* **Model Type:** CRNN (CNN + RNN + CTC),Paddleocr 2.6.1.3
* **Deployment:** Offline (Jetson / Linux / Windows)

---

## 🏋️ Training Pipeline

* Image normalization & augmentation
* Sequence labeling
* CTC loss–based training
* Character dictionary defined manually
* Checkpoint-based training & recovery

---

## 🔍 Inference Flow

1. Load input image
2. Detect text regions
3. Crop & preprocess regions
4. Run CRNN recognition
5. Decode sequence output
6. Validate fields
7. Display OK / NG with bounding boxes

---

## 🖥️ Offline GUI Validator

* Displays detected bounding boxes
* Shows recognized text per region
* Real-time **OK / NG** decision on image
* Designed for **shop-floor / inspection usage**

---

## 📦 Installation

1️⃣ Clone the Repository
[git clone (https://github.com/naman2328/GUI-OCR.git)
cd GUI-OCR

```bash
pip install -r requirements.txt
```

> ⚠️ This project is intended to run **offline**.
> No internet or cloud dependencies are required.

---

## ▶️ Run Inference

```bash
python inference/run_ocr.py --image sample.jpg
```

---

## 📊 Output Example


* Recognized text overlay
* Validation status (OK / NG)

---

## 🧩 Use Cases

* Industrial part marking verification
* Serial number & label validation
* Manufacturing quality inspection
* Offline OCR for compliance-sensitive environments

---

## 📌 Notes

* All models, logic, and pipelines are **originally implemented**
* No third-party OCR engines are used
* Designed for **extensibility and auditability**

---

## 👤 Author

**Naman Sharma**
Mechatronics Engineer
Focus: Robotics, Computer Vision

---
