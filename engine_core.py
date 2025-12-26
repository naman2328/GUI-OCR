import cv2
import os
import numpy as np
import time
from paddleocr import PaddleOCR, draw_ocr
from pyzbar.pyzbar import decode as decode_barcode
from pylibdmtx.pylibdmtx import decode as decode_datamatrix
from datetime import datetime
from difflib import SequenceMatcher

class Engine:
    def __init__(self, test_mode=True, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        self.test_mode = test_mode
        self.ocr = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.7,
            use_space_char=True,
            use_gpu=True,  # Set False if CPU only
        )
        self.font_path = font_path
        self.temp_dir = "ocr_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.expected_entries = [
            "CA","0080.P1.11.0022","AUTOLIV",
            "Avenue de l'Europe, B.P.99","UK","0589","CE",
            "France","2806","76220 Gournay-en-Bray","63932400",
            "AIRBAG MODULE DAB ADP-1","0080.P1.11.0022","AIRBAG"
        ]

    @staticmethod
    def normalize(text):
        return text.lower().replace(" ", "").replace(",", "").replace(".", "")

    def is_match(self, expected, detected_list, threshold=0.95):
        exp_norm = self.normalize(expected)
        for det in detected_list:
            ratio = SequenceMatcher(None, exp_norm, self.normalize(det)).ratio()
            if ratio >= threshold:
                return True
        return False

    @staticmethod
    def crop_from_box(image, box, padding=5):
        box = np.array(box).astype(int)
        x_min = max(min(pt[0] for pt in box) - padding, 0)
        y_min = max(min(pt[1] for pt in box) - padding, 0)
        x_max = min(max(pt[0] for pt in box) + padding, image.shape[1])
        y_max = min(max(pt[1] for pt in box) + padding, image.shape[0])
        return image[y_min:y_max, x_min:x_max]

    def run_cycle(self, frame):
        start_time = time.time()
        try:
            result = self.ocr.ocr(frame, cls=True)
            boxes, texts, scores, all_detected = [], [], [], []

            for idx, line in enumerate(result):
                for i, word_info in enumerate(line):
                    box = word_info[0]
                    text = word_info[1][0]
                    conf = word_info[1][1]
                    boxes.append(box)
                    texts.append(text)
                    scores.append(conf)
                    all_detected.append(text)

            for b in decode_barcode(frame):
                all_detected.append(b.data.decode("utf-8"))
            for d in decode_datamatrix(frame):
                all_detected.append(d.data.decode("utf-8"))

            matched = [item for item in self.expected_entries if self.is_match(item, all_detected)]
            not_matched = [item for item in self.expected_entries if item not in matched]
            status_text = "OK" if not not_matched else "NG"

            img_ocr = draw_ocr(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                boxes, texts, scores,
                font_path=self.font_path
            )
            img_ocr = cv2.cvtColor(img_ocr, cv2.COLOR_RGB2BGR)
            color = (0, 255, 0) if status_text == "OK" else (0, 0, 255)
            cv2.putText(img_ocr, f"STATUS: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            end_time = time.time()
            time_taken = end_time - start_time
            cv2.putText(img_ocr, f"Time: {time_taken:.2f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
            img_path = os.path.join(self.temp_dir, f"ocr_{timestamp}.png")
            cv2.imwrite(img_path, img_ocr)

            return {
                "status": status_text,
                "text": all_detected,
                "matched": matched,
                "not_matched": not_matched,
                "conf": np.mean(scores) if scores else 0,
                "path": img_path,
                "time_taken": time_taken
            }

        except Exception as e:
            return {"status": "ERROR", "text": [], "matched": [], "not_matched": [], "conf": 0, "path": "", "time_taken": 0}

