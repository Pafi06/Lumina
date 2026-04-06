"""
AstroStack — astrophotography stacking app
Run: python app.py
Deps: pip install PyQt6 numpy scipy astropy rawpy imageio pillow
      pip install scikit-image  (optional, for alignment)
"""

import sys, os
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
    QProgressBar, QScrollArea, QFrame, QGridLayout, QSizePolicy,
    QMessageBox, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QDragEnterEvent, QDropEvent

# ── image loading (multi-format) ──────────────────────────────────────────────

def load_img(p):
    """Load any image file into float32 numpy array, grayscale."""
    p = str(p)
    ext = Path(p).suffix.lower()

    # FITS
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        with fits.open(p) as h:
            d = h[0].data.astype(np.float32)
            if d.ndim == 3:
                d = d[0]  # take first channel if cube
            return d

    # RAW (CR2, NEF, ARW, DNG, RAF …)
    raw_exts = {".cr2",".cr3",".nef",".arw",".dng",".raf",".rw2",".orf",".pef",".srw"}
    if ext in raw_exts:
        import rawpy
        with rawpy.imread(p) as r:
            rgb = r.postprocess(
                use_camera_wb=True, half_size=False,
                no_auto_bright=True, output_bps=16
            ).astype(np.float32)
        return rgb.mean(axis=2)  # luminance

    # Everything else (JPG, PNG, TIFF, BMP …)
    from PIL import Image
    img = Image.open(p).convert("L")
    return np.array(img, dtype=np.float32)


# ── calibration ───────────────────────────────────────────────────────────────

def make_master(paths):
    if not paths:
        return None
    return np.median(np.array([load_img(p) for p in paths]), axis=0)

def calibrate(light, mb, md, mf):
    d = light.copy()
    if mb is not None: d -= mb
    if md is not None: d -= md
    if mf is not None:
        nf = mf / (np.median(mf) + 1e-9)
        nf = np.clip(nf, 0.01, None)
        d /= nf
    return d


# ── alignment ─────────────────────────────────────────────────────────────────

def align_frames(frames):
    try:
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import shift as nd_shift
    except ImportError:
        return frames, "skimage not installed — skipped alignment"

    ref = frames[0]
    out = [ref]
    for f in frames[1:]:
        sh, _, _ = phase_cross_correlation(ref, f, upsample_factor=10)
        out.append(nd_shift(f, sh))
    return out, None


# ── stacking ──────────────────────────────────────────────────────────────────

def sigma_clip(stack, n=3, iters=3):
    s = stack.copy()
    mask = np.zeros_like(s, dtype=bool)
    for _ in range(iters):
        m = np.ma.array(s, mask=mask)
        mu  = m.mean(axis=0).data
        sig = m.std(axis=0).data
        mask = np.abs(s - mu) > n * sig
    return np.ma.array(s, mask=mask).mean(axis=0).data

STACK_FN = {
    "Sigma Clip (recommended)": sigma_clip,
    "Median":  lambda s: np.median(s, axis=0),
    "Mean":    lambda s: np.mean(s, axis=0),
}


# ── preview stretch ───────────────────────────────────────────────────────────

def stretch(d, lo=0.1, hi=99.9):
    a, b = np.percentile(d, lo), np.percentile(d, hi)
    return np.clip((d - a) / (b - a + 1e-9), 0, 1)

def to_qpixmap(arr, max_w=600, max_h=500):
    """Convert float [0,1] 2D array to QPixmap."""
    u8 = (stretch(arr) * 255).astype(np.uint8)
    h, w = u8.shape
    qi = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
    px = QPixmap.fromImage(qi)
    return px.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)


# ── worker thread ─────────────────────────────────────────────────────────────

class Worker(QThread):
    progress = pyqtSignal(int, str)
    done     = pyqtSignal(object, str)  # result array, message
    error    = pyqtSignal(str)

    def __init__(self, lights, biases, darks, flats, method, do_align):
        super().__init__()
        self.lights  = lights
        self.biases  = biases
        self.darks   = darks
        self.flats   = flats
        self.method  = method
        self.do_align = do_align

    def run(self):
        try:
            n = len(self.lights)
            self.progress.emit(0, "Building calibration masters…")
            mb = make_master(self.biases)
            md = make_master(self.darks)
            mf = make_master(self.flats)

            frames = []
            for i, p in enumerate(self.lights):
                self.progress.emit(int((i+1)/n*50), f"Calibrating {Path(p).name}")
                raw = load_img(p)
                frames.append(calibrate(raw, mb, md, mf))

            if self.do_align:
                self.progress.emit(55, "Aligning frames…")
                frames, warn = align_frames(frames)
            
            self.progress.emit(70, f"Stacking ({self.method})…")
            fn = STACK_FN[self.method]
            result = fn(np.array(frames))

            self.progress.emit(100, "Done")
            self.done.emit(result, f"Stacked {n} frames")
        except Exception as e:
            self.error.emit(str(e))


# ── drop zone widget ──────────────────────────────────────────────────────────

class DropZone(QFrame):
    files_added = pyqtSignal(list)

    def __init__(self, label):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        self._files = []

        self.setStyleSheet("""
            DropZone {
                border: 1px solid #2a2d3a;
                border-radius: 8px;
                background: #12141c;
            }
            DropZone:hover {
                border-color: #4a6fa5;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        self.title = QLabel(label)
        self.title.setStyleSheet("color: #8892a4; font-size: 10px; letter-spacing: 1.5px; font-weight: 600;")
        lay.addWidget(self.title)

        self.count_lbl = QLabel("Drop files here or click +")
        self.count_lbl.setStyleSheet("color: #3d4455; font-size: 12px;")
        lay.addWidget(self.count_lbl)

        btn = QPushButton("+ Add files")
        btn.setFixedHeight(28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background: #1e2130; color: #6b7a99;
                border: 1px solid #2a2d3a; border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background: #252838; color: #a0b0cc; }
        """)
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select files", "",
            "Images (*.fits *.fit *.fts *.cr2 *.cr3 *.nef *.arw *.dng "
            "*.raf *.rw2 *.orf *.jpg *.jpeg *.png *.tiff *.tif *.bmp)"
        )
        if paths:
            self._add(paths)

    def _add(self, paths):
        self._files.extend(paths)
        n = len(self._files)
        self.count_lbl.setText(f"{n} file{'s' if n!=1 else ''} loaded")
        self.count_lbl.setStyleSheet("color: #5b8dd9; font-size: 12px;")
        self.files_added.emit(paths)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        self._add(paths)

    def clear(self):
        self._files = []
        self.count_lbl.setText("Drop files here or click +")
        self.count_lbl.setStyleSheet("color: #3d4455; font-size: 12px;")

    @property
    def files(self):
        return self._files


# ── main window ───────────────────────────────────────────────────────────────

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result = None
        self.worker = None
        self.setWindowTitle("AstroStack")
        self.setMinimumSize(1100, 720)
        self._build_ui()
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget#root {
                background: #0d0f18;
            }
            QLabel { color: #c8d0e0; }
            QComboBox {
                background: #12141c; color: #a0b0cc;
                border: 1px solid #2a2d3a; border-radius: 5px;
                padding: 4px 10px; font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1a1d2a; color: #a0b0cc;
                selection-background-color: #2a3a5a;
            }
            QCheckBox { color: #7a8aaa; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #2a2d3a; border-radius: 3px;
                background: #12141c;
            }
            QCheckBox::indicator:checked { background: #3a6ad4; border-color: #3a6ad4; }
            QProgressBar {
                background: #12141c; border: 1px solid #1e2130;
                border-radius: 4px; height: 6px; text-align: center;
                color: transparent;
            }
            QProgressBar::chunk { background: #3a6ad4; border-radius: 4px; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: #0d0f18; width: 6px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #2a2d3a; border-radius: 3px; min-height: 20px;
            }
        """)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ── left sidebar ──
        side = QWidget()
        side.setFixedWidth(300)
        side.setStyleSheet("background: #0a0c14; border-right: 1px solid #181b26;")
        sl = QVBoxLayout(side)
        sl.setContentsMargins(20, 24, 20, 20)
        sl.setSpacing(16)

        # logo
        logo = QLabel("ASTROSTACK")
        logo.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 18px; font-weight: 700; letter-spacing: 4px;
            color: #5b8dd9;
        """)
        sl.addWidget(logo)

        tagline = QLabel("astrophotography stacking")
        tagline.setStyleSheet("color: #3d4455; font-size: 10px; letter-spacing: 2px; margin-top: -10px;")
        sl.addWidget(tagline)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #181b26;")
        sl.addWidget(div)
        sl.addSpacing(4)

        # drop zones
        self.dz_lights = DropZone("LIGHTS")
        self.dz_darks  = DropZone("DARKS")
        self.dz_flats  = DropZone("FLATS")
        self.dz_biases = DropZone("BIASES")

        for dz in [self.dz_lights, self.dz_darks, self.dz_flats, self.dz_biases]:
            sl.addWidget(dz)

        sl.addSpacing(4)

        # options
        opt_lbl = QLabel("OPTIONS")
        opt_lbl.setStyleSheet("color: #3d4455; font-size: 10px; letter-spacing: 2px;")
        sl.addWidget(opt_lbl)

        self.method_cb = QComboBox()
        self.method_cb.addItems(STACK_FN.keys())
        sl.addWidget(self.method_cb)

        self.align_chk = QCheckBox("Align frames")
        sl.addWidget(self.align_chk)

        self.hotpix_chk = QCheckBox("Fix hot pixels")
        sl.addWidget(self.hotpix_chk)

        sl.addStretch()

        # progress
        self.prog = QProgressBar()
        self.prog.setValue(0)
        sl.addWidget(self.prog)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #3d4455; font-size: 11px;")
        sl.addWidget(self.status_lbl)

        # action buttons
        self.stack_btn = QPushButton("STACK")
        self.stack_btn.setFixedHeight(44)
        self.stack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stack_btn.setStyleSheet("""
            QPushButton {
                background: #1e3a7a; color: #7ab0ff;
                border: 1px solid #2a4a9a; border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 13px; font-weight: 700; letter-spacing: 3px;
            }
            QPushButton:hover { background: #254894; color: #a0c8ff; }
            QPushButton:pressed { background: #1a2e60; }
            QPushButton:disabled { background: #131520; color: #2a2d3a; border-color: #1a1d26; }
        """)
        self.stack_btn.clicked.connect(self._run)
        sl.addWidget(self.stack_btn)

        self.save_btn = QPushButton("Save Result")
        self.save_btn.setFixedHeight(34)
        self.save_btn.setEnabled(False)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #4a5a7a;
                border: 1px solid #1e2130; border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover { border-color: #3a4a6a; color: #7a9acc; }
            QPushButton:disabled { color: #1e2130; border-color: #141620; }
        """)
        self.save_btn.clicked.connect(self._save)
        sl.addWidget(self.save_btn)

        main.addWidget(side)

        # ── right panel — preview ──
        right = QWidget()
        right.setStyleSheet("background: #0d0f18;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(32, 28, 32, 28)
        rl.setSpacing(16)

        ph = QLabel("PREVIEW")
        ph.setStyleSheet("color: #2a2d3a; font-size: 10px; letter-spacing: 3px; font-family: 'Courier New';")
        rl.addWidget(ph)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("""
            background: #080a12;
            border: 1px solid #181b26;
            border-radius: 8px;
            color: #2a2d3a;
            font-size: 14px;
        """)
        self.preview.setText("Stack images to see preview")
        rl.addWidget(self.preview)

        # stats bar
        self.stats_lbl = QLabel("")
        self.stats_lbl.setStyleSheet("color: #3d4455; font-size: 11px; font-family: 'Courier New';")
        rl.addWidget(self.stats_lbl)

        main.addWidget(right)

    # ── run pipeline ──────────────────────────────────────────────────────────

    def _run(self):
        if not self.dz_lights.files:
            QMessageBox.warning(self, "No lights", "Add at least one light frame.")
            return

        self.stack_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.prog.setValue(0)

        self.worker = Worker(
            self.dz_lights.files,
            self.dz_biases.files,
            self.dz_darks.files,
            self.dz_flats.files,
            self.method_cb.currentText(),
            self.align_chk.isChecked(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.prog.setValue(pct)
        self.status_lbl.setText(msg)

    def _on_done(self, result, msg):
        self.result = result
        self.status_lbl.setText(msg)
        self.stack_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        # update preview
        px = to_qpixmap(result,
                        max_w=self.preview.width()-10,
                        max_h=self.preview.height()-10)
        self.preview.setPixmap(px)

        mn, mx, med = result.min(), result.max(), np.median(result)
        self.stats_lbl.setText(
            f"min {mn:.1f}   max {mx:.1f}   median {med:.1f}   "
            f"shape {result.shape[1]}×{result.shape[0]}"
        )

    def _on_error(self, msg):
        self.status_lbl.setText("Error")
        self.stack_btn.setEnabled(True)
        QMessageBox.critical(self, "Stack failed", msg)

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "stacked",
            "FITS (*.fits);;PNG (*.png);;TIFF (*.tiff)"
        )
        if not path:
            return

        ext = Path(path).suffix.lower()
        if ext in (".fits", ""):
            from astropy.io import fits
            if not path.endswith(".fits"):
                path += ".fits"
            fits.writeto(path, self.result.astype(np.float32), overwrite=True)
        else:
            from PIL import Image
            u8 = (stretch(self.result) * 255).astype(np.uint8)
            Image.fromarray(u8).save(path)

        self.status_lbl.setText(f"Saved → {Path(path).name}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.result is not None:
            px = to_qpixmap(self.result,
                            max_w=self.preview.width()-10,
                            max_h=self.preview.height()-10)
            self.preview.setPixmap(px)


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica Neue", 12))
    win = MainWin()
    win.show()
    sys.exit(app.exec())
