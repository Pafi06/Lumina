"""
Lumina — Professional Astrophotography Stacker
"""

import sys, os
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
    QProgressBar, QFrame, QSizePolicy, QMessageBox, QSpinBox,
    QGroupBox, QToolButton, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor

# --- Keep existing image processing logic (load_img, calibrate, etc.) ---
# [I've skipped repeating the heavy math functions to focus on your UI requests]
# [Ensure your original load_img, align_frames, and sigma_clip functions stay above this line]

# (Placeholder for the functions in your original script)
def load_img(p):
    from PIL import Image
    return np.array(Image.open(p).convert("L"), dtype=np.float32)

def guess_frame_type(path):
    n = Path(path).stem.lower()
    if any(x in n for x in ("flat","ff")): return "flat"
    if any(x in n for x in ("dark","dk")): return "dark"
    if any(x in n for x in ("bias","offset","bsf")): return "bias"
    return "light"

def linear_stretch(d, lo=0.1, hi=99.9):
    a, b = np.percentile(d, lo), np.percentile(d, hi)
    return np.clip((d - a) / (b - a + 1e-9), 0, 1)

def stf_stretch(d, target=0.25):
    lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
    nd = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    m = np.median(nd)
    if m < 1e-9: return nd
    b = (m - 1) / ((2*m - 1) * target + m - 1e-9) if abs(2*m-1) > 1e-9 else 0.5
    b = np.clip(b, 0.001, 0.999)
    return np.clip(((b - 1) * nd) / ((2*b - 1) * nd - b + 1e-9), 0, 1)

def to_qpixmap(arr, w, h, use_stf=False):
    s = stf_stretch(arr) if use_stf else linear_stretch(arr)
    u8 = (s * 255).astype(np.uint8)
    r, c = u8.shape
    qi = QImage(u8.data, c, r, c, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qi).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

# ─────────────────────────────────────────────────────────────────────────────
#  NEW DESIGNED DROP ZONE
# ─────────────────────────────────────────────────────────────────────────────

ZONE_COLORS = {"LIGHTS": "#60a5fa", "DARKS": "#a78bfa", "FLATS": "#fbbf24", "BIASES": "#34d399"}

class DropZone(QFrame):
    files_added = pyqtSignal(list)
    def __init__(self, label):
        super().__init__()
        self.label = label
        self._files = []
        col = ZONE_COLORS.get(label, "#60a5fa")
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)

        self.setStyleSheet(f"""
            DropZone {{
                border: 2px solid #1e293b; border-radius: 12px; background: #0f172a;
            }}
            DropZone:hover {{ border-color: {col}; }}
        """)

        lay = QVBoxLayout(self)
        hdr = QHBoxLayout()
        tl = QLabel(label)
        tl.setStyleSheet(f"color: {col}; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        hdr.addWidget(tl)
        hdr.addStretch()
        
        self.clear_btn = QToolButton()
        self.clear_btn.setText("✕")
        self.clear_btn.setStyleSheet("color: #64748b; border: none; font-weight: bold;")
        self.clear_btn.clicked.connect(self.clear)
        hdr.addWidget(self.clear_btn)
        lay.addLayout(hdr)

        self.count_lbl = QLabel("No files selected")
        self.count_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;") # Brighter text
        lay.addWidget(self.count_lbl)

        btn = QPushButton("Add")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"background: #1e293b; color: white; border-radius: 6px; padding: 4px; font-size: 11px;")
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, f"Add {self.label}", "")
        if paths: self._add(paths)

    def _add(self, paths):
        self._files.extend(paths)
        col = ZONE_COLORS.get(self.label, "#60a5fa")
        self.count_lbl.setText(f"{len(self._files)} files loaded")
        self.count_lbl.setStyleSheet(f"color: {col}; font-weight: bold;")
        self.files_added.emit(paths)

    def clear(self):
        self._files = []
        self.count_lbl.setText("No files selected")
        self.count_lbl.setStyleSheet("color: #94a3b8;")

    def dragEnterEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else None
    def dropEvent(self, e): self._add([u.toLocalFile() for u in e.mimeData().urls()])

# ─────────────────────────────────────────────────────────────────────────────
#  SMART DROP (REMOVED EMOJI)
# ─────────────────────────────────────────────────────────────────────────────

class SmartDropBanner(QFrame):
    sorted_files = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFixedHeight(50)
        self.setStyleSheet("border: 2px dashed #334155; border-radius: 10px; background: #020617;")
        lay = QHBoxLayout(self)
        lbl = QLabel("AUTO-SORT MODE: DROP MIXED FOLDER HERE")
        lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        lay.addStretch(); lay.addWidget(lbl); lay.addStretch()

    def dragEnterEvent(self, e): e.acceptProposedAction()
    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        buckets = {"light": [], "dark": [], "flat": [], "bias": []}
        for p in paths:
            if Path(p).is_dir():
                for f in Path(p).rglob("*"):
                    if f.is_file(): buckets[guess_frame_type(f)].append(str(f))
            else: buckets[guess_frame_type(p)].append(p)
        self.sorted_files.emit(buckets)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW REFACTOR
# ─────────────────────────────────────────────────────────────────────────────

LOGO_PATH = Path(__file__).parent / "lumina_logo.png"

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina")
        self.setMinimumSize(1200, 800)
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background: #020617;") # Darker, richer background
        self.setCentralWidget(root)
        ml = QHBoxLayout(root)
        ml.setContentsMargins(0,0,0,0)

        # --- Sidebar ---
        side_container = QWidget()
        side_container.setFixedWidth(320) # Wider sidebar for long text
        side_container.setStyleSheet("background: #0f172a; border-right: 2px solid #1e293b;")
        sl = QVBoxLayout(side_container)

        # Logo Section (Brightened & Bordered)
        hdr = QHBoxLayout()
        if LOGO_PATH.exists():
            logo_lbl = QLabel()
            px = QPixmap(str(LOGO_PATH)).scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(px)
            logo_lbl.setStyleSheet("border: 2px solid #38bdf8; border-radius: 8px; padding: 2px; background: #1e293b;")
            hdr.addWidget(logo_lbl)
        
        title = QLabel("LUMINA")
        title.setStyleSheet("font-size: 26px; font-weight: 800; color: #f8fafc; letter-spacing: 4px;")
        hdr.addWidget(title)
        hdr.addStretch()
        sl.addLayout(hdr)

        # Scrollable area for controls (prevent cutting off)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        scroll_content = QWidget()
        scroll_lay = QVBoxLayout(scroll_content)
        
        scroll_lay.addWidget(SmartDropBanner())
        self.dz_lights = DropZone("LIGHTS")
        self.dz_darks = DropZone("DARKS")
        self.dz_flats = DropZone("FLATS")
        self.dz_biases = DropZone("BIASES")
        for dz in [self.dz_lights, self.dz_darks, self.dz_flats, self.dz_biases]: scroll_lay.addWidget(dz)

        # Options
        opt = QGroupBox("PROCESSING")
        opt.setStyleSheet("QGroupBox { color: #38bdf8; font-weight: bold; border: 1px solid #334155; margin-top: 20px; }")
        ol = QVBoxLayout(opt)
        self.method_cb = QComboBox()
        self.method_cb.addItems(["Sigma Clip (recommended)", "Median", "Mean"])
        ol.addWidget(self.method_cb)
        
        self.align_chk = QCheckBox("Align Frames")
        self.grad_chk = QCheckBox("Remove Light Pollution") # Renamed for clarity
        for chk in [self.align_chk, self.grad_chk]:
            chk.setStyleSheet("color: #e2e8f0; font-weight: 500;")
            ol.addWidget(chk)
        
        scroll_lay.addWidget(opt)
        scroll_lay.addStretch()
        scroll.setWidget(scroll_content)
        sl.addWidget(scroll)

        # Footer Actions
        self.prog = QProgressBar()
        self.prog.setStyleSheet("QProgressBar { background: #1e293b; border: none; height: 6px; } QProgressBar::chunk { background: #38bdf8; }")
        sl.addWidget(self.prog)

        self.stack_btn = QPushButton("START STACKING")
        self.stack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stack_btn.setFixedHeight(50)
        self.stack_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb);
                color: white; font-weight: 800; font-size: 14px; border-radius: 10px; border: 1px solid #60a5fa;
            }
            QPushButton:hover { background: #60a5fa; }
        """)
        sl.addWidget(self.stack_btn)

        ml.addWidget(side_container)

        # --- Preview Area ---
        from __main__ import PreviewWidget # Ensure PreviewWidget is defined
        self.preview = PreviewWidget()
        ml.addWidget(self.preview)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Cleaner cross-platform look
    win = MainWin()
    win.show()
    sys.exit(app.exec())
