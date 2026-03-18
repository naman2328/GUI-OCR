# OCR Inspection System | Autoliv India Internship

---

## Overview

An offline OCR-based quality inspection system built during my internship at **Autoliv India (PED Department)**. Designed for manufacturing shop-floor deployment — no internet connection required.

The system reads text from industrial part labels, validates recognized content against expected values, and provides a real-time OK / NG decision through a shop-floor GUI.

---

## What It Does

- Detects and reads text from industrial part labels and markings
- Validates recognized text against configurable field rules (regex, length, format)
- Displays bounding boxes and recognized text in real-time
- Renders clear OK / NG status overlay for operator use
- Fully offline — no cloud dependency, deployable on Jetson or standard hardware

---

## System Architecture

```
Input Image
     │
     ▼
Text Detection (PaddleOCR)
     │  Bounding boxes per text region
     ▼
Text Recognition (PaddleOCR)
     │  Raw recognized strings
     ▼
Field Validation (Custom Logic)
     │  Regex / format / length rules per field
     ▼
OK / NG Decision
     │
     ▼
GUI Overlay (OpenCV)
     Bounding boxes + recognized text + status
```

---

## What I Built

PaddleOCR 2.6.1.3 is used as the core detection and recognition engine.

**Custom components built on top:**

- **Field Validation Logic** — configurable per-field rules using regex and format matching to produce OK / NG decisions
- **OK / NG Overlay** — visual status rendering on the original image for each detected field
- **Shop-Floor GUI** — operator-facing interface displaying bounding boxes, recognized text, and validation results in real time
- **Offline Deployment Configuration** — packaged for edge deployment on Jetson and standard Linux/Windows environments without internet dependency

---

## Engineering Note

Evaluated multiple OCR approaches before selecting PaddleOCR 2.6.1.3 as the recognition backbone based on its reliability and accuracy on industrial label data. Engineering effort was focused on the validation logic, GUI, and deployment pipeline — where the actual business value for the inspection use case was.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| OCR Engine | PaddleOCR 2.6.1.3 |
| Image Processing | OpenCV, NumPy |
| GUI | OpenCV / Tkinter |
| Deployment Target | Offline — Jetson / Linux / Windows |


---

## Installation

```bash
# Clone the repository
git clone https://github.com/naman2328/GUI-OCR.git
cd GUI-OCR

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ This system is designed for offline use. No internet connection required at runtime.

---

## Usage

```bash
python main.py --image sample.jpg
```

---

## Output

- Detected text regions with bounding boxes
- Recognized text displayed per region
- OK / NG validation status overlaid on image
- Console log of all field validation results

---

## Use Cases

- Industrial part marking verification
- Serial number and label validation
- Manufacturing quality inspection
- Offline OCR for compliance-sensitive environments

---

## Context

Built at **Autoliv India** as part of a computer vision and AI-focused internship in the PED (Product Engineering Department). The goal was to reduce manual inspection effort on the production line through automated text verification.

---

## Author

**Naman Sharma**
Mechatronics Engineer | Computer Vision | ROS 2 | Edge AI
[GitHub](https://github.com/naman2328) · [LinkedIn](www.linkedin.com/in/
namansharma4015
)
