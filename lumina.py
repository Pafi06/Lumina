"""
Lumina — astrophotography stacking app
Run:   py lumina.py
Deps:  py -m pip install PyQt6 numpy scipy astropy rawpy Pillow scikit-image
Logo:  lumina_logo.png  (place next to this script)
"""

import sys, os, re
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
    QProgressBar, QFrame, QSizePolicy, QMessageBox, QSlider,
    QTabWidget, QScrollArea, QGridLayout, QSpinBox, QDoubleSpinBox,
    QGroupBox, QToolButton, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QIcon, QPainter, QColor

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

RAW_EXT = {".cr2",".cr3",".nef",".arw",".dng",".raf",".rw2",".orf",".pef",".srw"}
FITS_EXT = {".fits",".fit",".fts"}

def load_img(p):
    p = str(p)
    ext = Path(p).suffix.lower()
    if ext in FITS_EXT:
        from astropy.io import fits
        with fits.open(p) as h:
            d = h[0].data.astype(np.float32)
            return d[0] if d.ndim == 3 else d
    if ext in RAW_EXT:
        import rawpy
        with rawpy.imread(p) as r:
            rgb = r.postprocess(use_camera_wb=True, no_auto_bright=True,
                                output_bps=16).astype(np.float32)
        return rgb.mean(axis=2)
    from PIL import Image
    return np.array(Image.open(p).convert("L"), dtype=np.float32)

def guess_frame_type(path):
    """Guess light/dark/flat/bias from filename."""
    n = Path(path).stem.lower()
    if any(x in n for x in ("flat","ff")): return "flat"
    if any(x in n for x in ("dark","dk")): return "dark"
    if any(x in n for x in ("bias","offset","bsf")): return "bias"
    return "light"

# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def make_master(paths):
    if not paths: return None
    return np.median(np.array([load_img(p) for p in paths]), axis=0)

def calibrate(light, mb, md, mf):
    d = light.copy()
    if mb is not None: d -= mb
    if md is not None: d -= md
    if mf is not None:
        nf = np.clip(mf / (np.median(mf) + 1e-9), 0.01, None)
        d /= nf
    return d

# ─────────────────────────────────────────────────────────────────────────────
#  ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def align_frames(frames):
    try:
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import shift as nd_shift
    except ImportError:
        return frames
    ref = frames[0]
    return [ref] + [nd_shift(f, phase_cross_correlation(ref, f, upsample_factor=10)[0])
                    for f in frames[1:]]

# ─────────────────────────────────────────────────────────────────────────────
#  STACKING
# ─────────────────────────────────────────────────────────────────────────────

def sigma_clip(stack, n=3, iters=3):
    s = stack.copy()
    mask = np.zeros_like(s, dtype=bool)
    for _ in range(iters):
        m = np.ma.array(s, mask=mask)
        mask = np.abs(s - m.mean(0).data) > n * m.std(0).data
    return np.ma.array(s, mask=mask).mean(0).data

STACK_FN = {
    "Sigma Clip (recommended)": sigma_clip,
    "Median":  lambda s: np.median(s, axis=0),
    "Mean":    lambda s: np.mean(s, axis=0),
}

# ─────────────────────────────────────────────────────────────────────────────
#  STRETCH  (linear + STF non-linear)
# ─────────────────────────────────────────────────────────────────────────────

def linear_stretch(d, lo=0.1, hi=99.9):
    a, b = np.percentile(d, lo), np.percentile(d, hi)
    return np.clip((d - a) / (b - a + 1e-9), 0, 1)

def stf_stretch(d, target=0.25):
    """PixInsight-style Screen Transfer Function (midtone stretch)."""
    lo = np.percentile(d, 0.5)
    hi = np.percentile(d, 99.5)
    nd = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    # midtone transfer: solve for b given target median
    m = np.median(nd)
    if m < 1e-9: return nd
    b = (m - 1) / ((2*m - 1) * target + m - 1e-9) if abs(2*m-1) > 1e-9 else 0.5
    b = np.clip(b, 0.001, 0.999)
    num = (b - 1) * nd
    den = (2*b - 1) * nd - b
    return np.clip(num / (den + 1e-9), 0, 1)

def to_qpixmap(arr, w, h, use_stf=False):
    s = stf_stretch(arr) if use_stf else linear_stretch(arr)
    u8 = (s * 255).astype(np.uint8)
    r, c = u8.shape
    qi = QImage(u8.data, c, r, c, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qi).scaled(
        w, h, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)

# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND EXTRACTION  (gradient / light pollution removal)
# ─────────────────────────────────────────────────────────────────────────────

def remove_gradient(d, degree=2):
    """
    Fit a 2D polynomial to the image background and subtract it.
    degree=1: plane removal   degree=2: curved gradient removal
    """
    h, w = d.shape
    # sample on a sparse grid to avoid fitting stars
    step = max(1, min(h, w) // 40)
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    vals = d[::step, ::step].ravel()

    # sigma-clip the samples to exclude bright stars
    med = np.median(vals)
    sig = vals.std()
    mask = np.abs(vals - med) < 2 * sig
    ys_s, xs_s, vs = ys.ravel()[mask], xs.ravel()[mask], vals[mask]

    # build polynomial features
    def poly_feats(y, x, deg):
        feats = []
        for i in range(deg+1):
            for j in range(deg+1-i):
                feats.append((y**i) * (x**j))
        return np.column_stack(feats)

    A = poly_feats(ys_s / h, xs_s / w, degree)
    coef, *_ = np.linalg.lstsq(A, vs, rcond=None)

    # evaluate on full grid
    yf, xf = np.mgrid[0:h, 0:w]
    Af = poly_feats(yf / h, xf / w, degree)
    bg = (Af @ coef).reshape(h, w)
    return d - bg

# ─────────────────────────────────────────────────────────────────────────────
#  WORKER
# ─────────────────────────────────────────────────────────────────────────────

class Worker(QThread):
    progress = pyqtSignal(int, str)
    done     = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, lights, biases, darks, flats,
                 method, do_align, do_hotpix, do_gradient, grad_deg):
        super().__init__()
        self.lights = lights; self.biases = biases
        self.darks = darks;   self.flats  = flats
        self.method = method; self.do_align = do_align
        self.do_hotpix = do_hotpix
        self.do_gradient = do_gradient; self.grad_deg = grad_deg

    def run(self):
        try:
            n = len(self.lights)
            self.progress.emit(2, "Building calibration masters…")
            mb = make_master(self.biases)
            md = make_master(self.darks)
            mf = make_master(self.flats)

            frames = []
            for i, p in enumerate(self.lights):
                self.progress.emit(int(5 + (i+1)/n*45), f"Calibrating {Path(p).name}")
                frames.append(calibrate(load_img(p), mb, md, mf))

            if self.do_hotpix:
                self.progress.emit(52, "Fixing hot pixels…")
                from scipy.ndimage import median_filter
                fixed = []
                for f in frames:
                    med = median_filter(f, 3)
                    mask = (f - med) > 5 * f.std()
                    fc = f.copy(); fc[mask] = med[mask]
                    fixed.append(fc)
                frames = fixed

            if self.do_align:
                self.progress.emit(58, "Aligning frames…")
                frames = align_frames(frames)

            self.progress.emit(70, f"Stacking ({self.method})…")
            result = STACK_FN[self.method](np.array(frames))

            if self.do_gradient:
                self.progress.emit(88, "Removing background gradient…")
                result = remove_gradient(result, self.grad_deg)

            self.progress.emit(100, f"Done — {n} frames stacked")
            self.done.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
#  DROP ZONE
# ─────────────────────────────────────────────────────────────────────────────

ZONE_COLORS = {
    "LIGHTS": "#4a8fff",
    "DARKS":  "#7a5af8",
    "FLATS":  "#f59e0b",
    "BIASES": "#10b981",
}

class DropZone(QFrame):
    files_added = pyqtSignal(list)

    def __init__(self, label, smart_cb=None):
        super().__init__()
        self.label = label
        self.smart_cb = smart_cb   # called with (path, guessed_type) for smart sort
        self.setAcceptDrops(True)
        self._files = []
        col = ZONE_COLORS.get(label, "#4a8fff")

        self.setStyleSheet(f"""
            DropZone {{
                border: 1px solid #1e2235;
                border-radius: 10px;
                background: #0e1020;
            }}
            DropZone:hover {{ border-color: {col}55; }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(3)

        hdr = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {col}; font-size: 8px;")
        hdr.addWidget(dot)
        tl = QLabel(label)
        tl.setStyleSheet(f"color: {col}; font-size: 10px; letter-spacing: 2px; font-weight: 700;")
        hdr.addWidget(tl)
        hdr.addStretch()

        self.clear_btn = QToolButton()
        self.clear_btn.setText("✕")
        self.clear_btn.setFixedSize(16, 16)
        self.clear_btn.setStyleSheet("color: #3d4455; border: none; font-size: 10px;")
        self.clear_btn.clicked.connect(self.clear)
        hdr.addWidget(self.clear_btn)
        lay.addLayout(hdr)

        self.count_lbl = QLabel("Drop files or click +")
        self.count_lbl.setStyleSheet("color: #2e3348; font-size: 11px;")
        lay.addWidget(self.count_lbl)

        btn = QPushButton("+ Add files")
        btn.setFixedHeight(24)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: #13162a; color: #3d4a6a;
                border: 1px solid #1e2235; border-radius: 4px; font-size: 10px;
            }}
            QPushButton:hover {{ color: {col}; border-color: {col}44; }}
        """)
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, f"Add {self.label}", "",
            "Images (*.fits *.fit *.fts *.cr2 *.cr3 *.nef *.arw *.dng "
            "*.raf *.rw2 *.orf *.jpg *.jpeg *.png *.tiff *.tif *.bmp)")
        if paths: self._add(paths)

    def _add(self, paths):
        self._files.extend(paths)
        n = len(self._files)
        col = ZONE_COLORS.get(self.label, "#4a8fff")
        self.count_lbl.setText(f"{n} file{'s' if n!=1 else ''}")
        self.count_lbl.setStyleSheet(f"color: {col}; font-size: 11px;")
        self.files_added.emit(paths)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        self._add([u.toLocalFile() for u in e.mimeData().urls()])

    def clear(self):
        self._files = []
        self.count_lbl.setText("Drop files or click +")
        self.count_lbl.setStyleSheet("color: #2e3348; font-size: 11px;")

    @property
    def files(self): return self._files

# ─────────────────────────────────────────────────────────────────────────────
#  SMART DROP ZONE  (receives any files, sorts automatically)
# ─────────────────────────────────────────────────────────────────────────────

class SmartDropBanner(QFrame):
    sorted_files = pyqtSignal(dict)   # emits {"light": [...], "dark": [...], ...}

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFixedHeight(48)
        self.setStyleSheet("""
            SmartDropBanner {
                border: 1px dashed #2a3050;
                border-radius: 8px;
                background: #0a0c1a;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        icon = QLabel("⚡")
        icon.setStyleSheet("font-size: 14px;")
        lay.addWidget(icon)
        lbl = QLabel("Smart drop — drop a mixed folder and files will be sorted automatically")
        lbl.setStyleSheet("color: #3d4a6a; font-size: 11px;")
        lay.addWidget(lbl)
        lay.addStretch()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            self.setStyleSheet("""
                SmartDropBanner {
                    border: 1px dashed #4a6fff;
                    border-radius: 8px;
                    background: #0d1025;
                }
            """)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.setStyleSheet("""
            SmartDropBanner {
                border: 1px dashed #2a3050;
                border-radius: 8px;
                background: #0a0c1a;
            }
        """)

    def dropEvent(self, e):
        self.dragLeaveEvent(e)
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        # expand directories
        all_files = []
        for p in paths:
            if Path(p).is_dir():
                for f in Path(p).rglob("*"):
                    if f.is_file(): all_files.append(str(f))
            else:
                all_files.append(p)
        # sort by guessed type
        buckets = {"light": [], "dark": [], "flat": [], "bias": []}
        for f in all_files:
            ext = Path(f).suffix.lower()
            ok_exts = FITS_EXT | RAW_EXT | {".jpg",".jpeg",".png",".tiff",".tif",".bmp"}
            if ext in ok_exts:
                buckets[guess_frame_type(f)].append(f)
        self.sorted_files.emit(buckets)

# ─────────────────────────────────────────────────────────────────────────────
#  PREVIEW WIDGET  with STF toggle + zoom
# ─────────────────────────────────────────────────────────────────────────────

class PreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._arr = None
        self._stf = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # toolbar
        tb = QHBoxLayout()

        hdr = QLabel("PREVIEW")
        hdr.setStyleSheet("color: #2a2d3a; font-size: 10px; letter-spacing: 3px; font-family: 'Courier New';")
        tb.addWidget(hdr)
        tb.addStretch()

        self.stf_btn = QPushButton("STF  OFF")
        self.stf_btn.setCheckable(True)
        self.stf_btn.setFixedHeight(26)
        self.stf_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stf_btn.setStyleSheet("""
            QPushButton {
                background: #0e1020; color: #3d4455;
                border: 1px solid #1e2235; border-radius: 5px;
                font-size: 10px; letter-spacing: 1px; padding: 0 12px;
                font-family: 'Courier New';
            }
            QPushButton:checked {
                background: #1a2a50; color: #4a8fff;
                border-color: #2a4a8f;
            }
        """)
        self.stf_btn.toggled.connect(self._toggle_stf)
        tb.addWidget(self.stf_btn)

        lay.addLayout(tb)

        # image label
        self.img_lbl = QLabel("Stack frames to see preview")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.img_lbl.setStyleSheet("""
            background: #06080f;
            border: 1px solid #141826;
            border-radius: 10px;
            color: #2a2d3a; font-size: 13px;
        """)
        lay.addWidget(self.img_lbl)

        self.stats_lbl = QLabel("")
        self.stats_lbl.setStyleSheet("color: #2e3348; font-size: 10px; font-family: 'Courier New';")
        lay.addWidget(self.stats_lbl)

    def set_data(self, arr):
        self._arr = arr
        self._refresh()
        mn, mx, med = arr.min(), arr.max(), np.median(arr)
        self.stats_lbl.setText(
            f"min {mn:.1f}  ·  max {mx:.1f}  ·  median {med:.1f}  ·  "
            f"{arr.shape[1]} × {arr.shape[0]} px")

    def _toggle_stf(self, on):
        self._stf = on
        self.stf_btn.setText("STF  ON" if on else "STF  OFF")
        self._refresh()

    def _refresh(self):
        if self._arr is None: return
        w = self.img_lbl.width() - 4
        h = self.img_lbl.height() - 4
        if w < 10 or h < 10: return
        px = to_qpixmap(self._arr, w, h, self._stf)
        self.img_lbl.setPixmap(px)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(50, self._refresh)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

LOGO_PATH = Path(__file__).parent / "lumina_logo.png"

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result = None
        self.worker = None
        self.setWindowTitle("Lumina")
        self.setMinimumSize(1180, 760)

        # window icon
        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self._build_ui()
        self._apply_theme()

    # ── theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #080a14; }
            QWidget { background: transparent; }
            QLabel { color: #c8d0e0; background: transparent; }
            QComboBox {
                background: #0e1020; color: #8899bb;
                border: 1px solid #1e2235; border-radius: 6px;
                padding: 5px 12px; font-size: 12px;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background: #0e1020; color: #8899bb;
                selection-background-color: #1a2a4a; border: 1px solid #1e2235;
            }
            QCheckBox { color: #6a7a9a; font-size: 12px; spacing: 10px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #2a2d3a; border-radius: 4px; background: #0e1020;
            }
            QCheckBox::indicator:checked { background: #3a6ad4; border-color: #4a7af4; }
            QProgressBar {
                background: #0a0c18; border: 1px solid #141826;
                border-radius: 4px; max-height: 5px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2a5ad4, stop:1 #4a8fff);
                border-radius: 4px;
            }
            QGroupBox {
                color: #3d4a6a; font-size: 10px; letter-spacing: 2px;
                border: 1px solid #141826; border-radius: 8px;
                margin-top: 8px; padding-top: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QSpinBox, QDoubleSpinBox {
                background: #0e1020; color: #8899bb;
                border: 1px solid #1e2235; border-radius: 5px; padding: 3px 8px;
            }
            QScrollBar:vertical { background: #080a14; width: 5px; }
            QScrollBar::handle:vertical { background: #1e2235; border-radius: 2px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background: #080a14;")
        self.setCentralWidget(root)
        ml = QHBoxLayout(root)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)

        # ══ SIDEBAR ══════════════════════════════════════════════════════════
        side = QWidget()
        side.setFixedWidth(290)
        side.setStyleSheet("background: #060810; border-right: 1px solid #0e1020;")
        sl = QVBoxLayout(side)
        sl.setContentsMargins(18, 20, 18, 18)
        sl.setSpacing(12)

        # logo + name
        hdr_row = QHBoxLayout()
        if LOGO_PATH.exists():
            logo_lbl = QLabel()
            px = QPixmap(str(LOGO_PATH)).scaled(
                44, 44,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(px)
            logo_lbl.setFixedSize(44, 44)
            hdr_row.addWidget(logo_lbl)
        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        name = QLabel("LUMINA")
        name.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 20px; font-weight: 700; letter-spacing: 5px; color: #4a8fff;
        """)
        tag = QLabel("astrophotography stacking")
        tag.setStyleSheet("color: #2a3050; font-size: 9px; letter-spacing: 2px;")
        name_col.addWidget(name)
        name_col.addWidget(tag)
        hdr_row.addLayout(name_col)
        hdr_row.addStretch()
        sl.addLayout(hdr_row)

        # divider
        div = QFrame(); div.setFixedHeight(1); div.setStyleSheet("background: #0e1020;")
        sl.addWidget(div)

        # smart drop banner
        self.smart_banner = SmartDropBanner()
        self.smart_banner.sorted_files.connect(self._on_smart_drop)
        sl.addWidget(self.smart_banner)

        # drop zones
        self.dz_lights = DropZone("LIGHTS")
        self.dz_darks  = DropZone("DARKS")
        self.dz_flats  = DropZone("FLATS")
        self.dz_biases = DropZone("BIASES")
        for dz in [self.dz_lights, self.dz_darks, self.dz_flats, self.dz_biases]:
            sl.addWidget(dz)

        # options group
        opt = QGroupBox("OPTIONS")
        ol = QVBoxLayout(opt)
        ol.setSpacing(8)

        self.method_cb = QComboBox()
        self.method_cb.addItems(STACK_FN.keys())
        ol.addWidget(self.method_cb)

        self.align_chk  = QCheckBox("Align frames")
        self.hotpix_chk = QCheckBox("Fix hot pixels")
        ol.addWidget(self.align_chk)
        ol.addWidget(self.hotpix_chk)

        # gradient removal
        grad_row = QHBoxLayout()
        self.grad_chk = QCheckBox("Remove gradient")
        grad_row.addWidget(self.grad_chk)
        self.grad_deg = QSpinBox()
        self.grad_deg.setRange(1, 4)
        self.grad_deg.setValue(2)
        self.grad_deg.setFixedWidth(46)
        self.grad_deg.setToolTip("Polynomial degree (1=plane, 2=curved, higher=aggressive)")
        grad_row.addWidget(self.grad_deg)
        deg_lbl = QLabel("deg")
        deg_lbl.setStyleSheet("color: #3d4455; font-size: 10px;")
        grad_row.addWidget(deg_lbl)
        ol.addLayout(grad_row)

        sl.addWidget(opt)
        sl.addStretch()

        # progress + status
        self.prog = QProgressBar(); self.prog.setValue(0)
        sl.addWidget(self.prog)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #2e3450; font-size: 10px;")
        sl.addWidget(self.status_lbl)

        # STACK button
        self.stack_btn = QPushButton("STACK")
        self.stack_btn.setFixedHeight(46)
        self.stack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stack_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #1a3a7a, stop:1 #1e2d6a);
                color: #5a9fff;
                border: 1px solid #2a4a9a; border-radius: 8px;
                font-family: 'Courier New'; font-size: 14px;
                font-weight: 700; letter-spacing: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #1e44aa, stop:1 #243580);
                color: #8abfff;
            }
            QPushButton:pressed { background: #101e4a; }
            QPushButton:disabled { background: #0a0c14; color: #1e2235; border-color: #0e1020; }
        """)
        self.stack_btn.clicked.connect(self._run)
        sl.addWidget(self.stack_btn)

        # Save button
        self.save_btn = QPushButton("Save Result")
        self.save_btn.setFixedHeight(32)
        self.save_btn.setEnabled(False)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #3d4a6a;
                border: 1px solid #141826; border-radius: 6px; font-size: 11px;
            }
            QPushButton:hover { border-color: #2a3a6a; color: #6a8acc; }
            QPushButton:disabled { color: #141826; border-color: #0e1020; }
        """)
        self.save_btn.clicked.connect(self._save)
        sl.addWidget(self.save_btn)

        ml.addWidget(side)

        # ══ PREVIEW PANEL ════════════════════════════════════════════════════
        self.preview = PreviewWidget()
        pw = QWidget()
        pw.setStyleSheet("background: #080a14;")
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(28, 24, 28, 24)
        pl.addWidget(self.preview)
        ml.addWidget(pw)

    # ── smart drop handler ────────────────────────────────────────────────────

    def _on_smart_drop(self, buckets):
        n = sum(len(v) for v in buckets.values())
        if n == 0:
            QMessageBox.information(self, "Nothing found",
                "No recognised image files found in the dropped folder.")
            return
        if buckets["light"]: self.dz_lights._add(buckets["light"])
        if buckets["dark"]:  self.dz_darks._add(buckets["dark"])
        if buckets["flat"]:  self.dz_flats._add(buckets["flat"])
        if buckets["bias"]:  self.dz_biases._add(buckets["bias"])
        self.status_lbl.setText(
            f"Smart sort: {len(buckets['light'])}L "
            f"{len(buckets['dark'])}D "
            f"{len(buckets['flat'])}F "
            f"{len(buckets['bias'])}B")

    # ── run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not self.dz_lights.files:
            QMessageBox.warning(self, "No lights", "Add at least one light frame.")
            return
        self.stack_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.prog.setValue(0)

        self.worker = Worker(
            self.dz_lights.files, self.dz_biases.files,
            self.dz_darks.files,  self.dz_flats.files,
            self.method_cb.currentText(),
            self.align_chk.isChecked(),
            self.hotpix_chk.isChecked(),
            self.grad_chk.isChecked(),
            self.grad_deg.value(),
        )
        self.worker.progress.connect(lambda p, m: (self.prog.setValue(p), self.status_lbl.setText(m)))
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, result):
        self.result = result
        self.stack_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.preview.set_data(result)

    def _on_error(self, msg):
        self.status_lbl.setText("Error — see details")
        self.stack_btn.setEnabled(True)
        QMessageBox.critical(self, "Stack failed", msg)

    # ── save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if self.result is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "stacked",
            "FITS (*.fits);;PNG (*.png);;TIFF (*.tiff)")
        if not path: return
        ext = Path(path).suffix.lower()
        if ext in (".fits", ""):
            from astropy.io import fits
            fits.writeto(path if path.endswith(".fits") else path+".fits",
                         self.result.astype(np.float32), overwrite=True)
        else:
            from PIL import Image
            u8 = (linear_stretch(self.result) * 255).astype(np.uint8)
            Image.fromarray(u8).save(path)
        self.status_lbl.setText(f"Saved → {Path(path).name}")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))
    win = MainWin()
    win.show()
    sys.exit(app.exec())
