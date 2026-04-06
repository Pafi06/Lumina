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

# calibration

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


# alignment


def align_frames(frames):
    try:
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import shift as nd_shift
    except ImportError:
        return frames
    ref = frames[0]
    return [ref] + [nd_shift(f, phase_cross_correlation(ref, f, upsample_factor=10)[0])
                    for f in frames[1:]]


# stacking

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

# stretch linear + STF non-linear

def linear_stretch(d, lo=0.1, hi=99.9):
    a, b = np.percentile(d, lo), np.percentile(d, hi)
    return np.clip((d - a) / (b - a + 1e-9), 0, 1)

def stf_stretch(d, target=0.25):
    """PixInsight-style Screen Transfer Function (midtone stretch)."""
    lo = np.percentile(d, 0.5)
    hi = np.percentile(d, 99.5)
    nd = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
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

# light pollution removal

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


# colors and drop zone

THEME_COLOR = "#38bdf8" 

class DropZone(QFrame):
    files_added = pyqtSignal(list)
    def __init__(self, label):
        super().__init__()
        self.label = label
        self._files = []
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)

        self.setStyleSheet(f"""
            DropZone {{
                border: 2px solid #1e293b; 
                border-radius: 12px; 
                background: #0f172a;
            }}
            DropZone:hover {{ 
                border-color: {THEME_COLOR}; 
            }}
        """)

        lay = QVBoxLayout(self)
        hdr = QHBoxLayout()
        tl = QLabel(label)
        # Unified label color
        tl.setStyleSheet(f"color: {THEME_COLOR}; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        hdr.addWidget(tl)
        hdr.addStretch()
        
        self.clear_btn = QToolButton()
        self.clear_btn.setText("✕")
        self.clear_btn.setStyleSheet("color: #64748b; border: none; font-weight: bold;")
        self.clear_btn.clicked.connect(self.clear)
        hdr.addWidget(self.clear_btn)
        lay.addLayout(hdr)

        self.count_lbl = QLabel("No files selected")
        self.count_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;") 
        lay.addWidget(self.count_lbl)

        btn = QPushButton("Add")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: #1e293b; 
                color: white; 
                border: 1px solid #334155;
                border-radius: 6px; 
                padding: 4px; 
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: #334155;
                border-color: {THEME_COLOR};
            }}
        """)
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, f"Add {self.label}", "")
        if paths: self._add(paths)

    def _add(self, paths):
        self._files.extend(paths)
        self.count_lbl.setText(f"{len(self._files)} files loaded")
        # Text turns bright when files are present
        self.count_lbl.setStyleSheet(f"color: {THEME_COLOR}; font-weight: bold;")
        self.files_added.emit(paths)

    def clear(self):
        self._files = []
        self.count_lbl.setText("No files selected")
        self.count_lbl.setStyleSheet("color: #94a3b8;")

    def dragEnterEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else None
    def dropEvent(self, e): self._add([u.toLocalFile() for u in e.mimeData().urls()])

# smart drop

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

# main window refractor

LOGO_PATH = Path(__file__).parent / "lumina_logo.png"

# preview

class PreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.raw_data = None
        self.use_stf = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # top bar
        hdr = QHBoxLayout()
        self.title = QLabel("IMAGE PREVIEW")
        self.title.setStyleSheet("color: #94a3b8; font-weight: bold; letter-spacing: 2px;")
        hdr.addWidget(self.title)
        hdr.addStretch()

        self.stf_btn = QPushButton("AUTO-STRETCH (STF) ON")
        self.stf_btn.setCheckable(True)
        self.stf_btn.setChecked(True)
        self.stf_btn.setFixedWidth(180)
        self.stf_btn.setStyleSheet("""
            QPushButton { background: #1e293b; color: #38bdf8; border: 1px solid #38bdf8; border-radius: 4px; padding: 5px; }
            QPushButton:checked { background: #38bdf8; color: #020617; }
        """)
        self.stf_btn.clicked.connect(self.toggle_stf)
        hdr.addWidget(self.stf_btn)
        layout.addLayout(hdr)

        # image display
        self.img_lbl = QLabel("Stacked result will appear here")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("color: #475569; font-style: italic; border: 2px dashed #1e293b; border-radius: 20px;")
        self.img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.img_lbl)

    def display(self, data):
        self.raw_data = data
        self.update_view()

    def toggle_stf(self):
        self.use_stf = self.stf_btn.isChecked()
        self.stf_btn.setText(f"AUTO-STRETCH (STF) {'ON' if self.use_stf else 'OFF'}")
        if self.raw_data is not None:
            self.update_view()

    def update_view(self):
        if self.raw_data is None: return
        px = to_qpixmap(self.raw_data, self.img_lbl.width(), self.img_lbl.height(), self.use_stf)
        self.img_lbl.setPixmap(px)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.raw_data is not None:
            QTimer.singleShot(50, self.update_view)

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina")
        self.setMinimumSize(1200, 800)
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background: #020617;")
        self.setCentralWidget(root)
        ml = QHBoxLayout(root)
        ml.setContentsMargins(0,0,0,0)

        # sidebar
        side_container = QWidget()
        side_container.setFixedWidth(320)
        side_container.setStyleSheet("background: #0f172a; border-right: 2px solid #1e293b;")
        sl = QVBoxLayout(side_container)

        # logo
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

        # options
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

        # footer actions
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

        # preview area
        self.preview = PreviewWidget()
        ml.addWidget(self.preview)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWin()
    win.show()
    sys.exit(app.exec())
