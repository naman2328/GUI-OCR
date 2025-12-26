# filename: continuous_roi_ocr_gui.py
import sys
import cv2
import os
import json
import time
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QStackedWidget, QTextEdit, QFileDialog, QListWidget,
    QListWidgetItem, QInputDialog, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap, QFont, QMouseEvent
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from engine_core import Engine
import logging
from logging.handlers import RotatingFileHandler
from enum import Enum


ROIS_CONFIG_FILENAME = "rois_config.json"
PREVIEW_PREFIX = "roi_preview_"


# ---------- Industrial System State ----------
class SystemState(Enum):
    INIT = "INIT"
    IDLE = "IDLE"
    READY = "READY"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"

logger = logging.getLogger("industrial_vision")
logger.setLevel(logging.INFO)

_handler = RotatingFileHandler(
    "vision_system.log",
    maxBytes=5_000_000,
    backupCount=5
)
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)



# -------- Helper: Worker thread for OCR so UI doesn't block ----------
class OCRWorker(QThread):
    finished = Signal(dict)  # emits the engine result dict

    def __init__(self, engine, frame):
        super().__init__()
        self.engine = engine
        self.frame = None if frame is None else frame.copy()

    def run(self):
        if self.frame is None:
            return
        try:
            result = self.engine.run_cycle(self.frame)
        except Exception as e:
            result = {
                'status': f'ERR: {e}',
                'matched': [],
                'not_matched': [],
                'conf': 0.0,
                'time_taken': 0.0,
                'path': ''
            }
        self.finished.emit(result)


# -------- Custom QLabel to capture mouse events over video widget ----------
class VideoLabel(QLabel):
    mousePressed = Signal(object)   # emits QPoint
    mouseMoved = Signal(object)     # emits QPoint
    mouseReleased = Signal(object)  # emits QPoint

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mousePressed.emit(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        self.mouseMoved.emit(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mouseReleased.emit(event.pos())


# ---------- Camera Thread ----------
class CameraThread(QThread):
    frameCaptured = Signal(object)

    def __init__(self, cam_id=0, width=1024, height=720):
        super().__init__()
        self.cam_id = cam_id
        self.running = True
        self.width = width
        self.height = height

    def run(self):
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while self.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    frame = cv2.resize(frame, (self.width, self.height))
                except Exception:
                    pass
                self.frameCaptured.emit(frame)
            self.msleep(30)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ---------- Program Mode (All ROI-related functionality moved here) ----------
class ProgramMode(QWidget):
    def __init__(self, shared_state, cam_w=640, cam_h=480):
        super().__init__()
        self.shared_state = shared_state
        self.cam_w = cam_w
        self.cam_h = cam_h

        # drawing state
        self.is_drawing = False
        self.roi_start = None
        self.roi_end = None

        self.latest_frame = None

        # UI
        layout = QVBoxLayout()
        layout.addWidget(QLabel("PROGRAM MODE – ROI Configuration (multiple ROIs supported)"))

        # Live preview for drawing
        self.video_label = VideoLabel("Program Mode Preview")
        self.video_label.setFixedSize(self.cam_w, self.cam_h)
        self.video_label.setStyleSheet("background: #1c1c1c; border-radius: 8px; border: 2px solid #444;")
        layout.addWidget(self.video_label)

        # Buttons
        buttons_row = QHBoxLayout()
        self.btn_draw_roi = QPushButton("Draw ROI")
        self.btn_save_roi = QPushButton("Save ROI (Add)")
        self.btn_clear_temp = QPushButton("Cancel Draw")
        self.btn_export_all = QPushButton("Export All ROIs JSON")
        for b in (self.btn_draw_roi, self.btn_save_roi, self.btn_clear_temp, self.btn_export_all):
            b.setStyleSheet("background-color:#2d89ef;color:white;border-radius:6px;padding:6px;font-weight:bold;")
        buttons_row.addWidget(self.btn_draw_roi)
        buttons_row.addWidget(self.btn_save_roi)
        buttons_row.addWidget(self.btn_clear_temp)
        buttons_row.addWidget(self.btn_export_all)
        layout.addLayout(buttons_row)

        # Left: list of ROIs; Right: preview + actions
        middle = QHBoxLayout()

        # ROI list
        self.roi_list = QListWidget()
        self.roi_list.setFixedWidth(300)
        middle.addWidget(self.roi_list)

        right_col = QVBoxLayout()
        self.info_label = QLabel("ROI: None selected")
        self.info_label.setFont(QFont("Arial", 10, QFont.Bold))
        right_col.addWidget(self.info_label)

        self.preview_label = QLabel("ROI Preview")
        self.preview_label.setFixedSize(320, 240)
        self.preview_label.setStyleSheet("background:#111;border-radius:6px;border:1px solid #333;")
        right_col.addWidget(self.preview_label)

        # ROI action buttons
        btn_row2 = QHBoxLayout()
        self.btn_select_roi = QPushButton("Set Active")
        self.btn_rename_roi = QPushButton("Rename")
        self.btn_delete_roi = QPushButton("Delete")
        self.btn_export_roi = QPushButton("Export Selected")
        for b in (self.btn_select_roi, self.btn_rename_roi, self.btn_delete_roi, self.btn_export_roi):
            b.setStyleSheet("background-color:#2d89ef;color:white;border-radius:6px;padding:6px;font-weight:bold;")
        btn_row2.addWidget(self.btn_select_roi)
        btn_row2.addWidget(self.btn_rename_roi)
        btn_row2.addWidget(self.btn_delete_roi)
        btn_row2.addWidget(self.btn_export_roi)
        right_col.addLayout(btn_row2)

        middle.addLayout(right_col)
        layout.addLayout(middle)
        layout.addStretch()
        self.setLayout(layout)

        # Connections
        self.btn_draw_roi.clicked.connect(self._start_draw_roi)
        self.btn_save_roi.clicked.connect(self._save_roi_from_current)
        self.btn_clear_temp.clicked.connect(self._cancel_draw)
        self.btn_export_all.clicked.connect(self._export_all_json)

        self.roi_list.currentRowChanged.connect(self._on_list_selection_changed)
        self.btn_select_roi.clicked.connect(self._set_active_for_selected)
        self.btn_delete_roi.clicked.connect(self._delete_selected)
        self.btn_rename_roi.clicked.connect(self._rename_selected)
        self.btn_export_roi.clicked.connect(self._export_selected_json)

        self.video_label.mousePressed.connect(self._on_mouse_press)
        self.video_label.mouseMoved.connect(self._on_mouse_move)
        self.video_label.mouseReleased.connect(self._on_mouse_release)

        # Load existing ROIs into list (shared_state is preloaded by MainWindow)
        self._reload_roi_list_ui()

    # Called by MainWindow on camera frames
    def on_frame(self, frame):
        self.latest_frame = frame
        display = frame.copy()

        # draw saved rois
        rois = self.shared_state.get('rois', [])
        for idx, r in enumerate(rois):
            coords = r.get('coords')
            if coords:
                x1, y1, x2, y2 = coords
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # draw index label
                cv2.putText(display, f"{idx}:{r.get('name')}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # temp drawing rect
        if self.is_drawing and self.roi_start and self.roi_end:
            sx, sy = self.roi_start
            ex, ey = self.roi_end
            cv2.rectangle(display, (sx, sy), (ex, ey), (255, 200, 0), 2)

        # convert and show
        try:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    # Drawing actions
    def _start_draw_roi(self):
        self.is_drawing = True
        self.roi_start = None
        self.roi_end = None
        self.info_label.setText("ROI: Drawing mode - click & drag on preview")

    def _cancel_draw(self):
        self.is_drawing = False
        self.roi_start = None
        self.roi_end = None
        self.info_label.setText("ROI: Draw canceled")
        # refresh display if frame exists
        if self.latest_frame is not None:
            self.on_frame(self.latest_frame)

    def _on_mouse_press(self, pos):
        if not self.is_drawing:
            return
        x = int(pos.x()); y = int(pos.y())
        x = max(0, min(self.cam_w - 1, x))
        y = max(0, min(self.cam_h - 1, y))
        self.roi_start = (x, y)
        self.roi_end = None

    def _on_mouse_move(self, pos):
        if not self.is_drawing or self.roi_start is None:
            return
        x = int(pos.x()); y = int(pos.y())
        x = max(0, min(self.cam_w - 1, x))
        y = max(0, min(self.cam_h - 1, y))
        self.roi_end = (x, y)
        if self.latest_frame is not None:
            self.on_frame(self.latest_frame)

    def _on_mouse_release(self, pos):
        if not self.is_drawing or self.roi_start is None:
            return
        x = int(pos.x()); y = int(pos.y())
        x = max(0, min(self.cam_w - 1, x))
        y = max(0, min(self.cam_h - 1, y))
        self.roi_end = (x, y)
        self.is_drawing = False
        self.info_label.setText("ROI: drawn — click 'Save ROI (Add)' to store it")
        if self.latest_frame is not None:
            self.on_frame(self.latest_frame)

    def _save_roi_from_current(self):
        if self.roi_start is None or self.roi_end is None:
            self.info_label.setText("ROI: Nothing to save — draw first")
            return
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            self.info_label.setText("ROI: Too small")
            return

        name, ok = QInputDialog.getText(self, "ROI name", "Enter ROI name:", text=f"ROI_{len(self.shared_state.get('rois',[]))+1}")
        if not ok:
            self.info_label.setText("ROI: Save canceled")
            return
        name = name.strip() or f"ROI_{len(self.shared_state.get('rois', []))+1}"

        # create previews directory if needed
        preview_name = f"{PREVIEW_PREFIX}{int(time.time())}.png"
        preview_path = os.path.join(os.getcwd(), preview_name)

        if self.latest_frame is not None:
            roi_img = self.latest_frame[y1:y2, x1:x2].copy()
            cv2.imwrite(preview_path, roi_img)
        else:
            roi_img = None
            preview_path = ""

        rois = self.shared_state.setdefault('rois', [])
        rois.append({
            "name": name,
            "coords": (int(x1), int(y1), int(x2), int(y2)),
            "preview_path": preview_path
        })
        # set this newly added as active
        self.shared_state['active_roi_index'] = len(rois) - 1

        self.roi_start = None
        self.roi_end = None
        self.info_label.setText(f"ROI: '{name}' saved and set active")
        self._reload_roi_list_ui()
        # show preview for new ROI
        if roi_img is not None:
            self._show_preview_image(roi_img)

    def _reload_roi_list_ui(self):
        self.roi_list.clear()
        rois = self.shared_state.get('rois', [])
        for idx, r in enumerate(rois):
            name = r.get('name', f"ROI_{idx}")
            coords = r.get('coords', (0, 0, 0, 0))
            item = QListWidgetItem(f"{idx}: {name} {coords}")
            self.roi_list.addItem(item)

        # if active index present, select it
        active = self.shared_state.get('active_roi_index')
        if active is not None and 0 <= active < self.roi_list.count():
            self.roi_list.setCurrentRow(active)
        else:
            self.roi_list.setCurrentRow(-1)
            self.preview_label.setPixmap(QPixmap())
            self.info_label.setText("ROI: None selected")

    def _on_list_selection_changed(self, row):
        if row < 0:
            self.preview_label.setPixmap(QPixmap())
            self.info_label.setText("ROI: None selected")
            return
        rois = self.shared_state.get('rois', [])
        if row >= len(rois):
            return
        r = rois[row]
        name = r.get('name')
        coords = r.get('coords')
        self.info_label.setText(f"ROI: {row}: '{name}' {coords}")
        # show preview if exists, else crop latest frame
        preview_path = r.get('preview_path', '')
        if preview_path and os.path.exists(preview_path):
            img = cv2.imread(preview_path)
            if img is not None:
                self._show_preview_image(img)
                return
        # fallback: crop from latest frame
        if self.latest_frame is not None and coords:
            x1, y1, x2, y2 = coords
            try:
                roi_img = self.latest_frame[y1:y2, x1:x2]
                if roi_img is not None and roi_img.size != 0:
                    self._show_preview_image(roi_img)
                    return
            except Exception:
                pass
        # else clear preview
        self.preview_label.setPixmap(QPixmap())

    def _show_preview_image(self, arr):
        try:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio)
            self.preview_label.setPixmap(pix)
        except Exception:
            pass

    def _set_active_for_selected(self):
        row = self.roi_list.currentRow()
        if row < 0:
            self.info_label.setText("ROI: no selection to set active")
            return
        self.shared_state['active_roi_index'] = row
        self.info_label.setText(f"ROI: set active -> {row}")
        # make sure RunMode and MainWindow read active index from shared_state

    def _delete_selected(self):
        row = self.roi_list.currentRow()
        if row < 0:
            self.info_label.setText("ROI: no selection to delete")
            return
        rois = self.shared_state.get('rois', [])
        if row >= len(rois):
            return
        confirm = QMessageBox.question(self, "Delete ROI", f"Delete ROI '{rois[row].get('name')}'?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        # remove preview file if exists
        preview_path = rois[row].get('preview_path', '')
        try:
            if preview_path and os.path.exists(preview_path):
                os.remove(preview_path)
        except Exception:
            pass
        rois.pop(row)
        # adjust active index
        active = self.shared_state.get('active_roi_index')
        if active is not None:
            if active == row:
                self.shared_state['active_roi_index'] = None
            elif active > row:
                self.shared_state['active_roi_index'] = active - 1
        self._reload_roi_list_ui()
        self.info_label.setText("ROI: deleted")

    def _rename_selected(self):
        row = self.roi_list.currentRow()
        if row < 0:
            self.info_label.setText("ROI: no selection to rename")
            return
        rois = self.shared_state.get('rois', [])
        old = rois[row].get('name', f"ROI_{row}")
        name, ok = QInputDialog.getText(self, "Rename ROI", "New name:", text=old)
        if not ok:
            self.info_label.setText("ROI: rename canceled")
            return
        name = name.strip() or old
        rois[row]['name'] = name
        self._reload_roi_list_ui()
        self.info_label.setText(f"ROI: renamed to '{name}'")

    def _export_selected_json(self):
        row = self.roi_list.currentRow()
        if row < 0:
            self.info_label.setText("ROI: no selection to export")
            return
        rois = self.shared_state.get('rois', [])
        roi = rois[row]
        fname, _ = QFileDialog.getSaveFileName(self, "Export ROI JSON", f"{roi.get('name')}.json", "JSON Files (*.json)")
        if not fname:
            return
        data = {
            "name": roi.get('name'),
            "coords": {"x1": int(roi['coords'][0]), "y1": int(roi['coords'][1]),
                       "x2": int(roi['coords'][2]), "y2": int(roi['coords'][3])},
            "frame_width": self.cam_w,
            "frame_height": self.cam_h,
            "preview_path": roi.get('preview_path', '')
        }
        try:
            with open(fname, 'w') as f:
                json.dump(data, f, indent=2)
            self.info_label.setText(f"ROI: exported to {fname}")
        except Exception as e:
            self.info_label.setText(f"ROI: export failed: {e}")

    def _export_all_json(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Export ROIs JSON", "rois_export.json", "JSON Files (*.json)")
        if not fname:
            return
        rois = self.shared_state.get('rois', [])
        export_list = []
        for r in rois:
            coords = r.get('coords') or (0, 0, 0, 0)
            export_list.append({
                "name": r.get('name'),
                "coords": {"x1": int(coords[0]), "y1": int(coords[1]), "x2": int(coords[2]), "y2": int(coords[3])},
                "preview_path": r.get('preview_path', '')
            })
        try:
            with open(fname, 'w') as f:
                json.dump({"rois": export_list, "frame_width": self.cam_w, "frame_height": self.cam_h}, f, indent=2)
            self.info_label.setText(f"ROIs: exported to {fname}")
        except Exception as e:
            self.info_label.setText(f"ROIs: export failed: {e}")


# ---------- Run Mode (execution & OCR) ----------
class RunMode(QWidget):
    def __init__(self, shared_state, cam_w=640, cam_h=478):
        super().__init__()
        self.shared_state = shared_state
        self.cam_w = cam_w
        self.cam_h = cam_h

        # engine for OCR
        self.engine = Engine(test_mode=True)

        self.current_frame = None
        self.continuous_ocr_enabled = False
        self.ocr_timer_interval_ms = 700
        self.ocr_timer = QTimer(self)
        self.ocr_timer.timeout.connect(self._on_ocr_timer)
        self.ocr_start_time = None
        self.ocr_timeout_sec = 2.0  # 24/7 safe
        



        # UI
        self.video_label = VideoLabel("Camera Feed")
        self.video_label.setFixedSize(self.cam_w, self.cam_h)
        self.video_label.setStyleSheet("background:#1c1c1c;border-radius:8px;border:2px solid #444;")

        self.ocr_label = QLabel("OCR Output")
        self.ocr_label.setFixedSize(640, 480)
        self.ocr_label.setStyleSheet("background:#1c1c1c;border-radius:8px;border:2px solid #444;")

        self.status_label = QLabel("STATUS: Waiting")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)

        self.matched_text = QTextEdit(); self.matched_text.setReadOnly(True); self.matched_text.setFixedHeight(120)
        self.matched_text.setStyleSheet("background:#111;color:#0f0;padding:5px;border-radius:5px;")
        self.not_matched_text = QTextEdit(); self.not_matched_text.setReadOnly(True); self.not_matched_text.setFixedHeight(120)
        self.not_matched_text.setStyleSheet("background:#111;color:#f00;padding:5px;border-radius:5px;")
        self.confidence_text = QLabel("Confidence: ---"); self.time_text = QLabel("Time: ---")

        self.cam_selector = QComboBox(); self.cam_selector.addItems(["0", "1", "2"])
        self.btn_program = QPushButton("Switch to Program Mode")
        self.btn_capture = QPushButton("Single Capture & OCR")
        self.btn_toggle_continuous = QPushButton("Start Continuous OCR")
        for b in (self.btn_program, self.btn_capture, self.btn_toggle_continuous):
            b.setStyleSheet("background-color:#2d89ef;color:white;border-radius:8px;padding:8px;font-weight:bold;")

        video_ocr_layout = QVBoxLayout()
        video_ocr_layout.addWidget(self.video_label, alignment=Qt.AlignTop)
        video_ocr_layout.addSpacing(10)
        video_ocr_layout.addWidget(self.ocr_label)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.status_label)
        right_panel.addWidget(QLabel("Matched (OK)")); right_panel.addWidget(self.matched_text)
        right_panel.addWidget(QLabel("Not Matched (NG)")); right_panel.addWidget(self.not_matched_text)
        right_panel.addWidget(self.confidence_text); right_panel.addWidget(self.time_text)
        right_panel.addWidget(self.cam_selector); right_panel.addWidget(self.btn_program)
        right_panel.addWidget(self.btn_capture); right_panel.addWidget(self.btn_toggle_continuous)
        right_panel.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_ocr_layout, 2)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

        # signals
        self.btn_capture.clicked.connect(self._single_capture_ocr)
        self.btn_toggle_continuous.clicked.connect(self._toggle_continuous_ocr)

        self._ocr_worker = None

    def on_frame(self, frame):
        self.current_frame = frame
        display_frame = frame.copy()
        # draw active ROI if exists
        active_index = self.shared_state.get('active_roi_index')
        rois = self.shared_state.get('rois', [])
        if active_index is not None and 0 <= active_index < len(rois):
            coords = rois[active_index].get('coords')
            if coords:
                x1, y1, x2, y2 = coords
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Active: {active_index}:{rois[active_index].get('name')}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        # convert/show
        try:
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def _toggle_continuous_ocr(self):
        if self.continuous_ocr_enabled:
            self._stop_continuous_ocr()
        else:
            # ensure an active ROI exists for continuous OCR; else warn
            if self.shared_state.get('active_roi_index') is None:
                self.status_label.setText("STATUS: No active ROI — continuous OCR requires active ROI")
                return
            self._start_continuous_ocr()

    def _start_continuous_ocr(self):
        if self.continuous_ocr_enabled:
            return
        self.continuous_ocr_enabled = True
        self.btn_toggle_continuous.setText("Stop PLC Continuous OCR")
        self.ocr_timer.start(self.ocr_timer_interval_ms)
        self.status_label.setText("STATUS: Continuous PLC OCR started (active ROI)")

    def _stop_continuous_ocr(self):
        if not self.continuous_ocr_enabled:
            return
        self.continuous_ocr_enabled = False
        self.btn_toggle_continuous.setText("Start PLC Continuous OCR")
        self.ocr_timer.stop()
        self.status_label.setText("STATUS: Continuous PLC OCR stopped")

    def _on_ocr_timer(self):
        if self.current_frame is None:
            return
        rois = self.shared_state.get('rois', [])
        idx = self.shared_state.get('active_roi_index')
        if idx is not None and 0 <= idx < len(rois):
            x1, y1, x2, y2 = rois[idx]['coords']
            frame_to_process = self.current_frame[y1:y2, x1:x2].copy()
        else:
            frame_to_process = self.current_frame.copy()

        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            return
        self.ocr_start_time = time.time()
        self.shared_state['system_state'] = SystemState.RUNNING

        self._ocr_worker = OCRWorker(self.engine, frame_to_process)
        self.shared_state['system_state'] = SystemState.RUNNING
        self._ocr_worker.finished.connect(self._on_ocr_result)
        self._ocr_worker.start()
        self.status_label.setText("STATUS: OCR in progress...")
        # At END of _on_ocr_timer()
        self._check_ocr_watchdog()


    def _single_capture_ocr(self):
        if self.current_frame is None:
            self.status_label.setText("STATUS: No frame available")
            return
        rois = self.shared_state.get('rois', [])
        idx = self.shared_state.get('active_roi_index')
        if idx is not None and 0 <= idx < len(rois):
            x1, y1, x2, y2 = rois[idx]['coords']
            frame_to_process = self.current_frame[y1:y2, x1:x2].copy()
            self.status_label.setText("Trigger")
        else:
            frame_to_process = self.current_frame.copy()
            self.status_label.setText("STATUS: Single OCR on full frame")

        if self._ocr_worker is not None and self._ocr_worker.isRunning():
            self.status_label.setText("STATUS: OCR already running — try again shortly")
            return
        self._ocr_worker = OCRWorker(self.engine, frame_to_process)
        self._ocr_worker.finished.connect(self._on_ocr_result)
        self._ocr_worker.start()
        self.shared_state['system_state'] = SystemState.READY


    def _on_ocr_result(self, result: dict):
        self.status_label.setText(f"STATUS: {result.get('status','DONE')}")
        matched = result.get('matched', []) or []
        not_matched = result.get('not_matched', []) or []
        self.matched_text.setPlainText("\n".join(matched))
        self.not_matched_text.setPlainText("\n".join(not_matched))
        try:
            conf = result.get('conf', 0.0) or 0.0
            self.confidence_text.setText(f"Confidence: {conf*100:.1f}%")
        except Exception:
            self.confidence_text.setText("Confidence: ---")
        try:
            t = result.get('time_taken', 0.0) or 0.0
            self.time_text.setText(f"Time: {t:.2f}s")
        except Exception:
            self.time_text.setText("Time: ---")

        out_path = result.get('path', '')
        if out_path and os.path.exists(out_path):
            self._show_ocr_image_from_path(out_path)
        else:
            rois = self.shared_state.get('rois', [])
            idx = self.shared_state.get('active_roi_index')
            if idx is not None and 0 <= idx < len(rois) and self.current_frame is not None:
                x1, y1, x2, y2 = rois[idx]['coords']
                roi_img = self.current_frame[y1:y2, x1:x2]
                self._show_ocr_image_from_array(roi_img)
            elif self.current_frame is not None:
                self._show_ocr_image_from_array(self.current_frame)

    def _show_ocr_image_from_path(self, path):
        ocr_img = cv2.imread(path)
        if ocr_img is None:
            return
        self._show_ocr_image_from_array(ocr_img)

    def _show_ocr_image_from_array(self, arr):
        try:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.ocr_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass
    def _check_ocr_watchdog(self):
       if self._ocr_worker and self._ocr_worker.isRunning():
          elapsed = time.time() - self.ocr_start_time
          if elapsed > self.ocr_timeout_sec:
            logger.warning("OCR timeout detected")
            self.status_label.setText("STATUS: OCR timeout, restarting engine")
            self._force_restart_engine()
    def _force_restart_engine(self):
        try:
            if self._ocr_worker:
             self._ocr_worker.terminate()
             self._ocr_worker.wait()
        except Exception:
         pass

        try:
            self.engine = Engine(test_mode=True)
            logger.info("OCR engine restarted safely")
            self.shared_state['system_state'] = SystemState.READY
        except Exception as e:
            logger.error("Engine restart failed", exc_info=True)
            self.shared_state['system_state'] = SystemState.ERROR


        


# ---------- Main Window (owns single camera thread and shared state) ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Vision GUI - Multi-ROI Program Mode")

        # Shared state
        self.shared_state = {
            'rois': [],                # list of {name, coords, preview_path}
            'active_roi_index': None,  # which ROI is active for RunMode
            'frame_width': 640,
            'frame_height': 480
        }
        self.system_state = SystemState.INIT
        logger.info("System INIT")
        self.shared_state['current_model'] = None
        self.shared_state['available_models'] = []
        self.shared_state['labels'] = {}



        # camera settings
        self.cam_w = 640
        self.cam_h = 480
        self.shared_state['frame_width'] = self.cam_w
        self.shared_state['frame_height'] = self.cam_h

        # Load ROIs from disk if present
        self._load_rois_from_disk()

        # Camera thread
        self.cam_thread = CameraThread(cam_id=0, width=self.cam_w, height=self.cam_h)
        self.cam_thread.frameCaptured.connect(self._on_camera_frame)
        self.cam_thread.start()
        self.system_state = SystemState.IDLE
        logger.info("Camera started, system IDLE")


        # Modes
        self.stack = QStackedWidget()
        self.program_mode = ProgramMode(shared_state=self.shared_state, cam_w=self.cam_w, cam_h=self.cam_h)
        self.run_mode = RunMode(shared_state=self.shared_state, cam_w=self.cam_w, cam_h=self.cam_h)
        self.stack.addWidget(self.run_mode)
        self.stack.addWidget(self.program_mode)

        # Navigation wiring
        self.run_mode.btn_program.clicked.connect(self.show_program)
        back_btn = QPushButton("Back to Run Mode")
        back_btn.setStyleSheet("background-color:#2d89ef;color:white;border-radius:8px;padding:8px;font-weight:bold;")
        back_btn.clicked.connect(self.show_run)
        self.program_mode.layout().addWidget(back_btn)

        # camera selector
        self.run_mode.cam_selector.currentTextChanged.connect(self._change_camera_id)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)
        self.resize(1200, 950)
        self.show()

    def _load_rois_from_disk(self):
        if not os.path.exists(ROIS_CONFIG_FILENAME):
            return
        try:
            with open(ROIS_CONFIG_FILENAME, 'r') as f:
                data = json.load(f)
            rois = data.get('rois', [])
            loaded = []
            for r in rois:
                coords = r.get('coords')
                # ensure coords as tuple of ints
                if isinstance(coords, dict):
                    coords_t = (int(coords.get('x1',0)), int(coords.get('y1',0)), int(coords.get('x2',0)), int(coords.get('y2',0)))
                elif isinstance(coords, (list, tuple)) and len(coords) == 4:
                    coords_t = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
                else:
                    coords_t = (0,0,0,0)
                loaded.append({
                    "name": r.get('name', 'ROI'),
                    "coords": coords_t,
                    "preview_path": r.get('preview_path', '') or r.get('preview', '')
                })
            self.shared_state['rois'] = loaded
            # active index if present
            ai = data.get('active_index')
            if isinstance(ai, int) and 0 <= ai < len(loaded):
                self.shared_state['active_roi_index'] = ai
        except Exception:
            # ignore load errors
            pass

    def _save_rois_to_disk(self):
        rois = self.shared_state.get('rois', [])
        export_list = []
        for r in rois:
            coords = r.get('coords', (0,0,0,0))
            export_list.append({
                "name": r.get('name'),
                "coords": {"x1": int(coords[0]), "y1": int(coords[1]), "x2": int(coords[2]), "y2": int(coords[3])},
                "preview_path": r.get('preview_path', '')
            })
        data = {
            "rois": export_list,
            "active_index": self.shared_state.get('active_roi_index'),
            "frame_width": self.cam_w,
            "frame_height": self.cam_h
        }
        try:
            with open(ROIS_CONFIG_FILENAME, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _change_camera_id(self, cam_id_text):
        try:
            cam_id = int(cam_id_text)
        except Exception:
            cam_id = 0
        try:
            self.cam_thread.stop()
        except Exception:
            pass
        self.cam_thread = CameraThread(cam_id=cam_id, width=self.cam_w, height=self.cam_h)
        self.cam_thread.frameCaptured.connect(self._on_camera_frame)
        self.cam_thread.start()

    def _on_camera_frame(self, frame):
        # keep shared frame (optional)
        self.shared_state['camera_frame'] = frame
        # forward to both modes
        try:
            self.program_mode.on_frame(frame)
        except Exception:
            pass
        try:
            self.run_mode.on_frame(frame)
        except Exception:
            pass

    def show_program(self):
        if self.system_state == SystemState.RUNNING:
         QMessageBox.warning(
            self,
            "System Busy",
            "Stop OCR before entering Program Mode"
          )
         return

        logger.info("Switched to PROGRAM mode")
        self.program_mode._reload_roi_list_ui()
        self.stack.setCurrentWidget(self.program_mode)


    def show_run(self):
     self.system_state = SystemState.READY
     logger.info("Switched to RUN mode")
     self.stack.setCurrentWidget(self.run_mode)


    def closeEvent(self, event):
        # Save ROIs before exit
        self._save_rois_to_disk()
        # stop camera thread
        try:
            self.cam_thread.stop()
        except Exception:
            pass
        event.accept()

    def _refresh_models(self):
       root = "recipes"
       os.makedirs(root, exist_ok=True)

       models = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        ]
       self.shared_state['available_models'] = models
       return models
    def _create_new_model(self, model_name):
        if self.system_state == SystemState.RUNNING:
           QMessageBox.warning(self, "Busy", "Stop OCR before creating model")
           return False

        base = os.path.join("recipes", model_name)
        if os.path.exists(base):
           QMessageBox.warning(self, "Exists", "Model already exists")
           return False

        os.makedirs(base, exist_ok=True)

    # empty recipe
        with open(os.path.join(base, "recipe.json"), "w") as f:
          json.dump({"rois": []}, f, indent=2)

    # empty labels
        with open(os.path.join(base, "labels.json"), "w") as f:
         json.dump({}, f, indent=2)

        self.shared_state['current_model'] = model_name
        self.shared_state['rois'] = []
        self.shared_state['labels'] = {}
        self.shared_state['active_roi_index'] = None

        logger.info(f"Created new model: {model_name}")
        return True
    def _save_current_model(self):
         model = self.shared_state.get('current_model')
         if not model:
             return

         base = os.path.join("recipes", model)
         os.makedirs(base, exist_ok=True)

    # save ROIs
         rois_out = []
         for r in self.shared_state.get('rois', []):
            x1, y1, x2, y2 = r['coords']
            rois_out.append({
            "name": r['name'],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
         })

         with open(os.path.join(base, "recipe.json"), "w") as f:
          json.dump({"rois": rois_out}, f, indent=2)

    # save labels
         with open(os.path.join(base, "labels.json"), "w") as f:
          json.dump(self.shared_state.get('labels', {}), f, indent=2)

         logger.info(f"Saved model: {model}")

    def _load_model(self, model_name):
        if self.system_state == SystemState.RUNNING:
            QMessageBox.warning(self, "Busy", "Stop OCR before changing model")
            return False

        base = os.path.join("recipes", model_name)
        try:
            with open(os.path.join(base, "recipe.json")) as f:
                recipe = json.load(f)
            with open(os.path.join(base, "labels.json")) as f:
                labels = json.load(f)
        except Exception:
            QMessageBox.critical(self, "Error", "Model files corrupted")
            return False

        rois = []
        for r in recipe.get("rois", []):
            rois.append({
            "name": r["name"],
            "coords": (r["x1"], r["y1"], r["x2"], r["y2"]),
            "preview_path": ""
            })

        self.shared_state['current_model'] = model_name
        self.shared_state['rois'] = rois
        self.shared_state['labels'] = labels
        self.shared_state['active_roi_index'] = 0 if rois else None

        logger.info(f"Loaded model: {model_name}")
        return True

         
 

# ---------- App Entry ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    try:
        sys.exit(app.exec())
    except SystemExit:
        try:
            window.cam_thread.stop()
        except Exception:
            pass
