# AstroStack

Astrophotography stacking app with a dark GUI.

## Install

```bash
pip install -r requirements.txt
```

> **rawpy** needs libraw. On Windows it just works via pip.  
> On Mac: `brew install libraw` first.  
> On Linux: `sudo apt install libraw-dev`

## Run

```bash
python app.py
```

## Supported formats

| Type | Extensions |
|------|-----------|
| FITS | .fits .fit .fts |
| RAW  | .cr2 .cr3 .nef .arw .dng .raf .rw2 .orf .pef |
| Standard | .jpg .jpeg .png .tiff .tif .bmp |

## Workflow

1. Drop **Lights** into the Lights zone (required)
2. Optionally add Darks, Flats, Biases
3. Choose stacking method (Sigma Clip recommended)
4. Toggle alignment if needed (requires scikit-image)
5. Hit **STACK**
6. Preview appears on the right
7. **Save Result** as FITS, PNG, or TIFF

## Stacking methods

- **Sigma Clip** — rejects outliers (satellites, cosmic rays). Best for most cases.
- **Median** — robust, no outlier rejection tuning needed
- **Mean** — fastest, use only if you have very few frames

## Alignment

Uses phase cross-correlation (sub-pixel). Works well for small drift.  
Requires `scikit-image`. If not installed, stacking still works without alignment.
