# Lumina

Astrophotography stacking app.

## Supported formats

| Type | Extensions |
|------|-----------|
| FITS | .fits .fit .fts |
| RAW  | .cr2 .cr3 .nef .arw .dng .raf .rw2 .orf .pef |
| Standard | .jpg .jpeg .png .tiff .tif .bmp |

## Stacking methods

- **Sigma Clip** — rejects outliers (satellites, etc). Best for most cases.
- **Median** — robust, no outlier rejection tuning needed
- **Mean** — fastest, use only if you have very few frames

## Alignment

Uses phase cross-correlation (sub-pixel). Works well for small drift.  
