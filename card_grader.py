#!/usr/bin/env python3
import sys, io, warnings
warnings.filterwarnings("ignore")   # suppress torch/easyocr deprecation noise
if sys.stdout is None:
    sys.stdout = io.StringIO()          # windowed exe — absorb print() silently
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = io.StringIO()
elif hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
Pokemon Card Pre-Grading Pipeline
==================================
Every card receives a probable PSA grade band across 3 measurable axes:
  1. Centering  (L/R and T/B border ratio)
  2. Corners    (whitening % + Shi-Tomasi sharpness, from research paper)
  3. Surface    (print-line artifact detection — ALWAYS flagged unverified
                 without raking-light photography)

Plus:
  - Holo/foil detection (saturation variance + sparkle pixels + brightness variance)
  - OCR identification (card name + number — REQUIRED; warns if not installed)
  - Remediation checklist for every card (what must be done before submitting)

GRADING AXES (approach from crimsonthinker/psa_pokemon_cards + research paper):
  centering_grade  → measurable from flat scan
  corner_grade     → approximated; requires 10x loupe confirmation
  surface_grade    → CANNOT be assessed without raking light; always unverified
  (edge_grade excluded — new cards; edges require edge-on macro photography anyway)

Overall probable grade = min(centering, corners) — surface excluded until verified.

PLASTIC SLEEVE HANDLING:
  If the outer border of the corrected image is uniformly pale (clear sleeve), the
  sleeve is stripped automatically and a note is added to remediation:
    "Centering/corner measurements taken after sleeve removal — re-verify on bare card."

PRECONDITIONS for reliable results:
  - Fixed overhead distance, diffuse even lighting, flat surface
  - Card (or card in sleeve) must fill ≥ 5% of the image
  - No harsh glare; raking light needed for surface/edge defect detection

USAGE:
  python card_grader.py --pdfdir "C:/scans/"          # all binder PDFs (3x3 grid/page)
  python card_grader.py --pdf "C:/scans/binder.pdf"   # single PDF
  python card_grader.py --imgdir "C:/scans/cards/"    # folder of individual card photos
  python card_grader.py --image "C:/scans/card.jpg"   # single image
  python card_grader.py --pdfdir "C:/scans/" --no-tcg # skip TCG price lookup

OUTPUT:
  card_grader_output/
    cards/              corrected card images (sleeve stripped if applicable)
    card_results.csv    one row per card: all metrics, grade band, remediation
    card_results.json

DEPENDENCIES (required):
  pip install opencv-python numpy Pillow requests pypdfium2

OCR (required for card identification — install ONE):
  pip install easyocr                  (self-contained, downloads ~100 MB model on first run)
  pip install pytesseract              + Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki

Approach references:
  - crimsonthinker/psa_pokemon_cards   (4-axis grading framework)
  - rthorst/mint_condition             (low-level feature importance for grading)
  - NickPiscitelli/pokemon-card-analyzer (border width centering measurement)
  - ScienceDirect automated corner grading paper (Shi-Tomasi + Hough intersection corners)
"""

import cv2
import numpy as np
from PIL import Image
import os, csv, json, argparse, re, time, logging, sqlite3, datetime, subprocess, shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — all thresholds here, never hardcoded below
# ══════════════════════════════════════════════════════════════════════════════

# — PSA sub-grade thresholds ————————————————————————————————————————————————
# Centering: worse-side % (e.g. if left=63 right=37, worse=63)
#   PSA 10 ≤ 55/45   PSA 9 ≤ 60/40   PSA 8 ≤ 65/35   PSA 7 ≤ 70/30
CENTERING_GRADE_THRESHOLDS  = {10: 55.0, 9: 60.0, 8: 65.0, 7: 70.0}

# Corners: maximum whitening % across all 4 corners
#   PSA 10 ≤ 5%   PSA 9 ≤ 12%   PSA 8 ≤ 20%   PSA 7 ≤ 30%
CORNER_GRADE_THRESHOLDS     = {10: 5, 9: 12, 8: 20, 7: 30}

# — Centering scan ——————————————————————————————————————————————————————————
# Saturation-based: border (white/silver/metallic) has S < BORDER_SAT_MAX,
# card content (coloured frame/artwork) has S > BORDER_SAT_MAX.
# More robust than brightness thresholding which fails on dark silver borders
# (treats silver as non-white) and bright yellow artwork (treats it as border).
CENTERING_BORDER_SAT_MAX    = 50    # HSV saturation above this = card content (not border)
CENTERING_SCAN_SAMPLES      = 10    # scanlines averaged per axis

# — Corner analysis ————————————————————————————————————————————————————————
CORNER_CROP_PX              = 60    # corner crop size after correction
CORNER_WHITENING_DELTA      = 25    # brightness above local background = whitened pixel
CORNER_SHARPNESS_BLUR_K     = 5     # Gaussian blur kernel for Shi-Tomasi preprocessing
CORNER_SHARPNESS_QUALITY    = 0.01  # Shi-Tomasi quality level
CORNER_SHARPNESS_MIN_DIST   = 5     # Shi-Tomasi min distance between corners

# — Holo / foil detection ——————————————————————————————————————————————————
# Measured in the artwork region (rows 8–60%, cols 5–95% of corrected card).
# Signals calibrated against Bulbapedia reference scans (cosmos, tinsel, starlight,
# reverse holo) vs a confirmed non-holo card. Two of three signals must fire.
#
# sat_grad: mean Sobel magnitude on the S channel. Holo foil has fine, rapid
#   saturation changes (the dot/stripe pattern) → high gradient. Smooth artwork
#   colour washes → low gradient. Most reliable single discriminator.
#   Non-holo reference: ~72  |  Holo references: 84–182  |  User scan: 156
HOLO_SAT_GRAD_MIN           = 90.0  # non-holo <80, holo >84 — threshold set with margin
#
# local_sat_std: mean std computed over 16×16 patches. Holo has many patches with
#   internal texture; non-holo artwork has smooth gradients within patches.
#   Non-holo reference: ~26  |  Holo references: 32–48  |  User scan: 41
HOLO_LOCAL_SAT_STD_MIN      = 27.0  # non-holo ~22-26, holo 28–48
#
# sparkle_pct: % artwork pixels with V>235 (very bright). Works when light catches
#   the foil at the right angle; low threshold because overhead light is weak for this.
HOLO_SPARKLE_PCT_MIN        = 0.4   # even a few glint spots confirm foil

# — Print line / surface detection —————————————————————————————————————————
PRINT_LINE_MIN_LENGTH       = 80
PRINT_LINE_MAX_GAP          = 8
PRINT_LINE_HOUGH_THRESHOLD  = 40
PRINT_LINE_FLAG_COUNT       = 3

# — Card geometry ——————————————————————————————————————————————————————————
TARGET_CARD_W               = 630   # 63mm × 10 px/mm
TARGET_CARD_H               = 880   # 88mm × 10 px/mm

# — Card detection ————————————————————————————————————————————————————————
MIN_CARD_AREA_FRACTION      = 0.05
CONTOUR_APPROX_EPSILON      = 0.02

# — Scanner background removal (for uncropped flatbed scans) ———————————————
# Flatbed scanner background is typically RGB(248–255). Cards are slightly
# dimmer even when white. Lower this value if cards are not being detected.
SCANNER_BG_THRESHOLD        = 228   # pixels brighter than this = scanner background (LiDE 300 bg ≈ 232-238)
SCANNER_MORPH_CLOSE_PX      = 80    # morphological close kernel to fill card's white border

# — Sleeve detection ——————————————————————————————————————————————————————
SLEEVE_OUTER_PCT            = 0.06  # outer strip fraction to check for sleeve
SLEEVE_MEAN_MIN             = 185   # outer strip mean brightness to flag as clear plastic
SLEEVE_STD_MAX              = 30    # outer strip std deviation to flag as uniform
SLEEVE_COLOUR_SPREAD_MAX    = 35    # max brightness spread across all 4 strips for coloured sleeve
SLEEVE_COLOUR_JUMP          = 25    # brightness above sleeve mean that marks card white border

# — Binder page grid ——————————————————————————————————————————————————————
BINDER_ROWS                 = 3
BINDER_COLS                 = 3
BINDER_MARGIN_PCT           = 0.02
BINDER_CELL_PAD_PCT         = 0.03

# — PDF rendering ——————————————————————————————————————————————————————————
PDF_RENDER_SCALE            = 3.0

# — OCR ————————————————————————————————————————————————————————————————————
TESSERACT_CMD   = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Name region starts at 15% x to skip "BASIC"/"STAGE 1" evolution label at left
OCR_NAME_REGION = (0.15, 0.01, 0.72, 0.10)
OCR_NUM_REGION  = (0.03, 0.90, 0.40, 0.97)
OCR_HP_REGION   = (0.60, 0.01, 0.92, 0.09)
OCR_SCALE       = 4

# — Portfolio database ————————————————————————————————————————————————————
DB_PATH         = Path(r"C:\Users\User\Documents\pokemon-card-grader\card_grader_portfolio.db")

# — Paths ——————————————————————————————————————————————————————————————————
BASE_DIR        = Path(r"C:\Users\User\Pictures\poke card grader")
INPUT_DIR       = BASE_DIR                                   # legacy
SCANNING_DIR    = BASE_DIR / "scanning"                      # --scan saves here
REPORT_DROP_DIR = BASE_DIR / "report_drop"                   # drag good scan here
IMAGES_DIR      = BASE_DIR / "images"                        # compressed card images (kept)
OUTPUT_DIR      = BASE_DIR / "output"                        # CSV / DB
CSV_NAME        = "card_results.csv"
JSON_NAME       = "card_results.json"

# ══════════════════════════════════════════════════════════════════════════════
# SCANNER PROFILES
# Keyed by substring that appears in the WIA device name.
# scan_dpi   — DPI sent to the scanner (optical/2 for CIS; avoids interpolation)
# color_wia  — WIA 2.0 Data Type: 3 = colour
# scale      — internal card resolution multiplier (1.0 = 630×880, 2.0 = 1260×1760)
#              Scales CORNER_CROP_PX and PRINT_LINE_MIN_LENGTH proportionally.
# zone_mm    — (width_mm, height_mm) scan zone anchored at top-left of bed.
#              Card is expected in the top-left corner; None = full bed.
# ══════════════════════════════════════════════════════════════════════════════
_PY32      = Path(r"C:\Users\User\AppData\Local\py32_twain\python\python.exe")
_TWAIN_BRG = Path(r"C:\Users\User\AppData\Local\py32_twain\twain_bridge.py")

SCANNER_PROFILES: Dict[str, dict] = {
    "LiDE 300": {
        "model":       "Canon CanoScan LiDE 300",
        "sensor":      "CIS",
        "optical_dpi": 2400,
        "scan_dpi":    1200,         # via TWAIN / Canon ScanGear (32-bit bridge)
        "twain_dpi":   1200,         # TWAIN target DPI; absent → WIA fallback
        "color_wia":   3,            # WIA 2.0 colour (used for WIA fallback only)
        "scale":       2.0,          # 1260×1760 internal at 1200 DPI
        "zone_mm":     (189, 264),   # scan zone = 3× standard card (63×88 mm)
        "zone_center": True,         # centre zone on scanner bed
        "bed_mm":      (216, 297),   # LiDE 300 A4 bed physical dimensions
        "tip":         "Place card face-down in the CENTRE of the glass (art side touching glass). "
                       "Remove lid and scan in a dark room for best holo texture capture.",
    },
}

_DEFAULT_PROFILE: dict = {
    "model": "Unknown Scanner", "sensor": "unknown",
    "optical_dpi": 600, "scan_dpi": 600, "color_wia": 3,
    "scale": 1.0, "zone_mm": None, "tip": "",
}

# Active profile — overwritten by _detect_scanner() at startup
_SCANNER: dict = _DEFAULT_PROFILE.copy()

# ══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
OCR_ENGINE = None

def _init_ocr():
    global OCR_ENGINE
    try:
        import easyocr
        reader = easyocr.Reader(["en", "ja"], verbose=False)
        OCR_ENGINE = ("easyocr", reader)
        logging.info("OCR engine: easyocr (en+ja)")
        return
    except Exception as e:
        logging.debug(f"easyocr unavailable: {e}")
    try:
        import pytesseract
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        pytesseract.get_tesseract_version()
        OCR_ENGINE = ("tesseract", pytesseract)
        logging.info("OCR engine: pytesseract")
        return
    except Exception:
        pass
    print(
        "\n*** NO OCR ENGINE — card names will not be read. Fix with: ***\n"
        "  pip install easyocr\n"
        "  OR: pip install pytesseract + Tesseract-OCR from\n"
        "      https://github.com/UB-Mannheim/tesseract/wiki\n"
    )

try:
    import pypdfium2 as pdfium
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CenteringResult:
    left_pct:        float = 0.0
    right_pct:       float = 0.0
    top_pct:         float = 0.0
    bottom_pct:      float = 0.0
    centering_grade: int   = 10

    @property
    def lr_ratio(self) -> str:
        return f"{self.left_pct:.0f}/{self.right_pct:.0f}"

    @property
    def tb_ratio(self) -> str:
        return f"{self.top_pct:.0f}/{self.bottom_pct:.0f}"


@dataclass
class CornerResult:
    tl_whitening: float = 0.0   # % pixels above local background threshold
    tr_whitening: float = 0.0
    br_whitening: float = 0.0
    bl_whitening: float = 0.0
    tl_sharpness: float = 0.0   # Shi-Tomasi corner response at corner tip (higher = sharper)
    tr_sharpness: float = 0.0
    br_sharpness: float = 0.0
    bl_sharpness: float = 0.0
    corner_grade: int   = 10

    def whitening_scores(self) -> List[float]:
        return [self.tl_whitening, self.tr_whitening, self.br_whitening, self.bl_whitening]

    def worst_whitening(self) -> float:
        return max(self.whitening_scores())

    def sharpness_scores(self) -> List[float]:
        return [self.tl_sharpness, self.tr_sharpness, self.br_sharpness, self.bl_sharpness]


@dataclass
class HoloResult:
    is_holo:       bool  = False
    confidence:    str   = "none"    # "none" / "low" / "medium" / "high"
    sat_grad:      float = 0.0       # mean Sobel gradient on S channel (primary signal)
    local_sat_std: float = 0.0       # mean patch-level saturation std (secondary signal)
    sparkle_pct:   float = 0.0       # % artwork pixels with V>235 (tertiary signal)


@dataclass
class PrintLineResult:
    line_count: int  = 0
    flagged:    bool = False
    coords:     List = field(default_factory=list)


@dataclass
class GradeAssessment:
    """
    PSA grading axes + overall probable grade + remediation checklist.
    surface_grade is ALWAYS marked 'unverified' without raking-light photography.
    """
    centering_grade:  int  = 0
    corner_grade:     int  = 0
    surface_grade:    str  = "unverified"   # cannot measure without raking light
    probable_grade:   int  = 0             # min(centering, corner)
    grade_band:       str  = ""
    limiting_factor:  str  = ""
    surface_verified: bool = False
    is_holo:          bool = False
    remediation:      List[str] = field(default_factory=list)


@dataclass
class IDResult:
    card_name:          str   = ""
    card_number:        str   = ""
    set_name:           str   = ""
    set_code:           str   = ""
    rarity:             str   = ""
    hp:                 str   = ""
    ocr_raw_name:       str   = ""
    ocr_raw_number:     str   = ""
    ocr_confidence:     str   = "none"


@dataclass
class CardResult:
    filename:           str                       = ""
    card_index:         int                       = 0
    centering:          Optional[CenteringResult] = None
    corners:            Optional[CornerResult]    = None
    holo:               Optional[HoloResult]      = None
    print_lines:        Optional[PrintLineResult] = None
    grade:              Optional[GradeAssessment] = None
    identification:     Optional[IDResult]        = None
    corrected_img_path: str                       = ""
    error:              str                       = ""

# ══════════════════════════════════════════════════════════════════════════════
# CARD DETECTION & PERSPECTIVE CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def _find_best_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Strategy 1: Canny edge detection.
    Works best when the card has a visible coloured border against the background.
    Returns ordered [TL, TR, BR, BL] or None.
    """
    h, w     = bgr.shape[:2]
    min_area = w * h * MIN_CARD_AREA_FRACTION
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    best_cnt = None
    # Try progressively lower thresholds to find the subtle card-to-whitespace edge
    for lo, hi in [(30, 120), (15, 60), (8, 30)]:
        edges  = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges  = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:10]:
            if cv2.contourArea(cnt) < min_area:
                break
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, CONTOUR_APPROX_EPSILON * peri, True)
            if len(approx) == 4:
                return _order_corners(approx)
        # Track best large contour across all threshold passes for fallback
        if contours and cv2.contourArea(contours[0]) >= min_area:
            if best_cnt is None or cv2.contourArea(contours[0]) > cv2.contourArea(best_cnt):
                best_cnt = contours[0]
    # minAreaRect fallback: rotation-aware, only if no quad was found above
    if best_cnt is not None:
        rect = cv2.minAreaRect(best_cnt)
        box  = cv2.boxPoints(rect)
        return _order_corners(box)
    return None


def _find_card_scanner(bgr: np.ndarray, bg_mean: float = 235.0) -> Optional[np.ndarray]:
    """
    Strategy 2: Brightness threshold for uncropped flatbed scanner images.
    Threshold is set just below the measured scanner bg level so the card's
    silver/metallic border is captured as card pixels.
    Returns ordered [TL, TR, BR, BL] or None.
    """
    h, w     = bgr.shape[:2]
    min_area = w * h * MIN_CARD_AREA_FRACTION
    max_area = w * h * 0.50   # card can't fill more than 50 % of the scan zone
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive: stay just below measured scanner bg so bright metallic borders
    # are captured as card pixels.  Never exceed SCANNER_BG_THRESHOLD.
    threshold = min(SCANNER_BG_THRESHOLD, max(215, int(bg_mean) - 5))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Fill the card's white inner border so the whole card is one solid region
    k      = np.ones((SCANNER_MORPH_CLOSE_PX, SCANNER_MORPH_CLOSE_PX), np.uint8)
    filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    logging.info(f"_find_card_scanner: thr={threshold} area={int(area)} max={int(max_area)}")
    if area < min_area or area > max_area:
        logging.warning(f"_find_card_scanner: area {int(area)} outside [{int(min_area)},{int(max_area)}] — rejected")
        return None

    hull = cv2.convexHull(largest)
    rect = cv2.minAreaRect(hull)
    bw, bh = rect[1]
    # Reject if bounding box is more than 10 % larger than expected card dimensions.
    # Spurious scanner-noise pixels can extend the convex hull far beyond the card
    # even when the contour area check passes.
    # Expected card dims = scan zone / 3  (189mm = 3×63mm, 264mm = 3×88mm).
    exp_w, exp_h = w / 3.0, h / 3.0
    if min(bw, bh) > exp_w * 1.10 or max(bw, bh) > exp_h * 1.10:
        logging.warning(f"_find_card_scanner: hull {int(bw)}×{int(bh)} too large vs expected {int(exp_w)}×{int(exp_h)} — rejected")
        return None
    box  = cv2.boxPoints(rect)
    return _order_corners(box)


def _find_card_colourband(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Strategy 3: Detect the card's coloured type-band via HSV saturation, then
    extrapolate outward to estimate card edges. Fallback for very white cards on
    white scanner beds where brightness threshold also struggles.
    Returns ordered [TL, TR, BR, BL] or None.
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1]

    # Find all meaningfully coloured pixels (the type-band + artwork)
    _, sat_mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    k = np.ones((10, 10), np.uint8)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Bounding box of all coloured regions
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    x, y, cw, ch = cv2.boundingRect(all_pts)
    if cw < 20 or ch < 20:
        return None

    # The coloured band is inset from the card edge by the white border (~8% each side).
    # Extrapolate outward. Bottom text box adds ~25% below the artwork.
    pad_x = int(cw * 0.10)
    pad_y = int(ch * 0.10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + int(ch * 0.30))   # text box extends below artwork

    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype="float32")


def _rotate_and_crop(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Deskew a flat card (scanner image) using rotation only.
    Rules:
      - NO perspective warp, NO pixel stretching/distortion
      - NEVER crop into the card — always include the full card
      - Canvas expands if needed so card edges are never clipped
      - Only a scale-down resize at the end (LANCZOS4 for quality)

    Steps:
      1. minAreaRect → rotation angle
      2. Expand canvas so rotated card fits without clipping
      3. warpAffine rotation only (INTER_LANCZOS4)
      4. Tight crop around the card in the rotated image (with 2px safety margin)
      5. Resize to TARGET dimensions
    """
    h, w  = bgr.shape[:2]
    rect  = cv2.minAreaRect(corners.astype(np.float32))
    angle = rect[2]
    box_w, box_h = rect[1]

    # Ensure portrait orientation (taller than wide for a Pokemon card)
    if box_w > box_h:
        angle  += 90
        box_w, box_h = box_h, box_w
    logging.info(f"_rotate_and_crop: angle={angle:.1f}° box={box_w:.0f}×{box_h:.0f} ar={box_w/box_h:.3f}")

    # Expand canvas so the entire rotated image fits (prevents any corner clipping)
    rad     = abs(np.radians(angle))
    new_w   = int(w * abs(np.cos(rad)) + h * abs(np.sin(rad))) + 2
    new_h   = int(w * abs(np.sin(rad)) + h * abs(np.cos(rad))) + 2
    offset_x = (new_w - w) / 2
    offset_y = (new_h - h) / 2

    # Rotation around the centre of the EXPANDED canvas
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    M[0, 2] += offset_x
    M[1, 2] += offset_y

    rotated = cv2.warpAffine(
        bgr, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Transform the card centre through the same matrix
    card_cx, card_cy = rect[0]
    pt     = np.array([[card_cx, card_cy, 1.0]]).T
    new_pt = (M @ pt).flatten()
    ncx    = int(new_pt[0])
    ncy    = int(new_pt[1])

    # 4px safety margin — keep tight so corner analysis isn't polluted by background
    bw2    = int(box_w / 2) + 4
    bh2    = int(box_h / 2) + 4
    x1     = max(0, ncx - bw2)
    y1     = max(0, ncy - bh2)
    x2     = min(new_w, ncx + bw2)
    y2     = min(new_h, ncy + bh2)

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        crop = rotated

    return cv2.resize(crop, (TARGET_CARD_W, TARGET_CARD_H), interpolation=cv2.INTER_LANCZOS4)


def _warp_to(bgr: np.ndarray, corners: np.ndarray,
             out_w: int = TARGET_CARD_W, out_h: int = TARGET_CARD_H) -> np.ndarray:
    """Full perspective warp — only use for hand-held photos with 3D angle."""
    dst = np.array([[0,0],[out_w,0],[out_w,out_h],[0,out_h]], dtype="float32")
    M   = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4)


def _find_card_bbox_projection(bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """
    Find the card's exact bounding box using two-pass row/column statistics.

    Pass 1 — left/right edges:
      For 20 sample rows across mid-60 % of image height, scan inward from each
      side pixel-by-pixel.  A pixel is scanner bg if gray ∈ [LO, HI].  Card content
      (artwork, colored frame) is clearly below LO.  Use median of hits.

    Pass 2 — top/bottom edges:
      Using x1..x2 from pass 1, compute per-row mean AND std of gray across that
      x-strip.  Scanner bg: mean ≈ 235, std ≈ 1-2.  Silver CRI border: mean ≈ 228-232,
      std ≈ 4-8 (iridescent, non-uniform).  Card content: mean << 230, std >> 5.
      A row is "card" if mean < BG_MEAN_MAX OR std > BG_STD_MAX.  Scan from top/bottom.

    Returns (x1, y1, x2, y2) in full-scan pixels, or None if not card-sized.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    LO, HI         = 225.0, 243.0   # scanner bg brightness range
    BG_MEAN_MAX     = 232.0          # row mean above this = scanner bg (not card)
    BG_STD_MAX      = 3.5            # row std below this AND mean > BG_MEAN_MAX = bg

    # ── Pass 1: left / right edges ──────────────────────────────────────────
    sample_rows = np.linspace(int(h * 0.25), int(h * 0.75), 20, dtype=int)
    left_hits, right_hits = [], []
    for r in sample_rows:
        row = gray[r, :]
        bg  = (row >= LO) & (row <= HI)
        nz  = np.where(~bg)[0]
        left_hits.append(int(nz[0])  if len(nz) else w)
        nz_r = np.where(~bg[::-1])[0]
        right_hits.append(int(nz_r[0]) if len(nz_r) else w)

    x1 = max(0, int(np.median(left_hits))  - 5)
    x2 = min(w, w - int(np.median(right_hits)) + 5)
    if x2 - x1 < 100:
        logging.warning("_find_card_bbox_projection: pass-1 left/right failed")
        return None

    # ── Pass 2: top / bottom edges using row statistics ─────────────────────
    # Work on the card x-strip only to amplify the border vs bg contrast.
    # Scanner bg: mean ≈ 235, std ≈ 1-2.
    # Silver CRI border: mean ≈ 228-232 (just below BG_MEAN_MAX), std ≈ 4-8.
    # Card content: mean << 230 or std >> 4.
    strip     = gray[:, x1:x2]                        # (h, card_width)
    row_mean  = strip.mean(axis=1)                     # (h,)
    row_std   = strip.std(axis=1)                      # (h,)
    is_card   = (row_mean < BG_MEAN_MAX) | (row_std > BG_STD_MAX)

    # Find the largest contiguous run of card rows — rejects isolated noisy bg rows
    # that scatter CIS sensor noise can cause.
    best_start, best_end, cur_start, cur_len, best_len = 0, 0, 0, 0, 0
    for i, v in enumerate(is_card):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len  = cur_len
                best_start = cur_start
                best_end   = i
        else:
            cur_len = 0

    if best_len < 50:   # must have at least 50 consecutive card rows
        logging.warning(f"_find_card_bbox_projection: no contiguous card rows (best={best_len})")
        return None
    y1 = max(0, best_start - 5)
    y2 = min(h, best_end   + 5)

    cw, ch = x2 - x1, y2 - y1
    logging.info(
        f"_find_card_bbox_projection: x1={x1} x2={x2} y1={y1} y2={y2} → {cw}×{ch} "
        f"(mean_range=[{row_mean[y1:y2].min():.0f},{row_mean[y1:y2].max():.0f}] "
        f"std_range=[{row_std[y1:y2].min():.1f},{row_std[y1:y2].max():.1f}])"
    )

    # Must be within 35 % of expected card dimensions (scan zone = 3 × card)
    exp_w, exp_h = w / 3.0, h / 3.0
    if not (exp_w * 0.65 < cw < exp_w * 1.35 and exp_h * 0.65 < ch < exp_h * 1.35):
        logging.warning(
            f"_find_card_bbox_projection: {cw}×{ch} not card-sized (~{int(exp_w)}×{int(exp_h)}) — rejected"
        )
        return None
    return x1, y1, x2, y2


def perspective_correct(bgr: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
    """
    Find card boundary and straighten to TARGET_CARD_W × TARGET_CARD_H.

    Detection strategy depends on image source:

    Scanner images (pure-white border detected):
      1. Brightness threshold + convex hull  → finds true outer silver border
      2. Canny fallback (inner frame found)  → large expand to reach outer edge

    Photos (non-white background):
      1. Canny quad at multiple thresholds   → finds largest visible boundary
      2. Brightness threshold fallback

    All paths use rotation-only crop — never perspective warp — to preserve
    border proportions for centering measurement.
    """
    h_img, w_img = bgr.shape[:2]
    gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detect flatbed scanner: use the 4 image corners (50×50px).
    # Full edge strips are unreliable when the card is near an edge — a card at the
    # bottom of the scan zone would darken the bottom strip and flip is_scanner False.
    # Corners are always scanner background regardless of where the card is placed.
    c = 50
    corner_px = np.concatenate([
        gray_img[:c, :c].ravel(),          # top-left
        gray_img[:c, -c:].ravel(),         # top-right
        gray_img[-c:, :c].ravel(),         # bottom-left
        gray_img[-c:, -c:].ravel(),        # bottom-right
    ])
    b_mean = float(np.mean(corner_px))
    b_std  = float(np.std(corner_px))
    is_scanner = bool(b_mean > 215 and b_std < 20)
    logging.info(f"is_scanner={is_scanner} (corner mean={b_mean:.1f} std={b_std:.1f})")

    def _expand(c: np.ndarray, pct: float) -> np.ndarray:
        cx = float(np.mean(c[:, 0]))
        cy = float(np.mean(c[:, 1]))
        e  = c.astype(float).copy()
        e[:, 0] = np.clip(cx + (e[:, 0] - cx) * (1 + pct), 0, w_img - 1)
        e[:, 1] = np.clip(cy + (e[:, 1] - cy) * (1 + pct), 0, h_img - 1)
        return e.astype("float32")

    if is_scanner:
        # Brightness threshold traces the true outer card edge (silver border).
        # Validate aspect ratio: if the detected region is not card-shaped (e.g., only
        # the diagonal inner artwork is found because the silver border is too bright),
        # fall through to Canny so we don't rotate by a wild angle.
        corners = _find_card_scanner(bgr, b_mean)
        if corners is not None and _card_aspect_ok(corners):
            logging.info("perspective_correct: scanner brightness path")
            return _rotate_and_crop(bgr, _expand(corners, 0.08)), True, True
        if corners is not None:
            logging.warning("perspective_correct: scanner brightness found non-card shape — trying projection")

        # Method 2: row/column projection (finds exact card edge pixel-by-pixel).
        # Scans from each of the 4 scan edges inward until it leaves scanner bg
        # (mean outside [225,243] OR std > 5 over a 10-sample window).  Works for
        # CRI metallic cards — silver border has std > 5 even at similar brightness.
        bbox = _find_card_bbox_projection(bgr)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            adj = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2],
            ], dtype="float32")
            logging.info(f"perspective_correct: projection bbox → exact crop {x2-x1}×{y2-y1}")
            return _rotate_and_crop(bgr, adj), True, True

        # Method 3: Canny fallback.  Size guard: reject full-scan result.
        corners = _find_best_quad(bgr)
        if corners is not None and _card_aspect_ok(corners):
            rect   = cv2.minAreaRect(corners.astype(np.float32))
            bw, bh = rect[1]
            if bw * bh < h_img * w_img * 0.3:
                logging.info("perspective_correct: scanner Canny fallback (+11 % expand)")
                return _rotate_and_crop(bgr, _expand(corners, 0.11)), True, True
            logging.warning("perspective_correct: Canny result covers full scan area — no card detected")
    else:
        # Photo: Canny first (handles perspective distortion from handheld shots)
        corners = _find_best_quad(bgr)
        if corners is not None and _card_aspect_ok(corners):
            logging.info("perspective_correct: photo Canny path")
            return _rotate_and_crop(bgr, _expand(corners, 0.012)), True, False
        corners = _find_card_scanner(bgr, b_mean)
        if corners is not None and _card_aspect_ok(corners):
            logging.info("perspective_correct: photo brightness fallback")
            return _rotate_and_crop(bgr, _expand(corners, 0.002)), True, False

    # Colour band extrapolation — last resort for very white cards on white beds
    corners = _find_card_colourband(bgr)
    if corners is not None and _card_aspect_ok(corners):
        rect   = cv2.minAreaRect(corners.astype(np.float32))
        bw, bh = rect[1]
        if bw * bh < h_img * w_img * 0.7:
            return _rotate_and_crop(bgr, _expand(corners, 0.012)), True, is_scanner
        logging.warning("perspective_correct: colourband result covers full scan area — no card detected")

    logging.warning("Card boundary not found — cropping to 90% of image centre")
    ph = int(h_img * 0.05)
    pw = int(w_img * 0.05)
    crop = bgr[ph:h_img-ph, pw:w_img-pw]
    return cv2.resize(crop, (TARGET_CARD_W, TARGET_CARD_H), interpolation=cv2.INTER_LANCZOS4), False, is_scanner


def _card_aspect_ok(corners: np.ndarray) -> bool:
    """Return True if corners enclose a shape with card-like aspect ratio (63:88 ±30%)."""
    rect = cv2.minAreaRect(corners.astype(np.float32))
    bw, bh = rect[1]
    if bw <= 0 or bh <= 0:
        return False
    ar = bw / bh
    # Accept portrait (0.716) or landscape (1.397) with ±30 % tolerance
    return (0.50 < ar < 0.93) or (1.07 < ar < 1.83)


def _orient_score(img: np.ndarray) -> float:
    """
    Score for correct card orientation.  A Pokémon card's text box (white,
    bottom 35–40 %) is the most reliable landmark: it exists on every card and
    is the largest bright region in the lower portion.
    Score = fraction of white pixels in bottom 38 % minus fraction in top 20 %.
    Higher score → this orientation is more likely correct.
    """
    h, w   = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    col_lo = int(w * 0.08)
    col_hi = int(w * 0.92)
    bot = gray[int(h * 0.62):, col_lo:col_hi]
    top = gray[:int(h * 0.20), col_lo:col_hi]
    return float(np.mean(bot > 225)) - float(np.mean(top > 225))


def auto_orient(corrected: np.ndarray, scanner: bool = True) -> np.ndarray:
    """
    Detect right-side-up vs upside-down (0° vs 180° only).
    Card is always placed portrait face-down on the scanner — 90° rotations
    never occur, so only 0° and 180° are tested to avoid axis swaps in centering.
    """
    score_0   = _orient_score(corrected)
    rotated   = cv2.rotate(corrected, cv2.ROTATE_180)
    score_180 = _orient_score(rotated)
    logging.info(f"auto_orient: 0°={score_0:.3f}  180°={score_180:.3f}  → {'180°' if score_180 > score_0 else '0°'}")
    return rotated if score_180 > score_0 else corrected


# ══════════════════════════════════════════════════════════════════════════════
# PLASTIC SLEEVE / PROTECTOR DETECTION + STRIPPING
# ══════════════════════════════════════════════════════════════════════════════
def detect_and_strip_sleeve(corrected: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    """
    Detect if the perspective-corrected image contains a plastic sleeve/protector.
    Handles two sleeve types:
      - Clear/penny sleeves: outer strip is uniformly pale (mean > SLEEVE_MEAN_MIN)
      - Coloured/matte sleeves (KMC, Dragon Shield): outer strip is uniformly dark/coloured
        (detected by low std across all 4 sides with consistent colour)

    If sleeve detected:
      - Scan inward from each edge to find the card's white border
      - Strategy A (clear sleeve): HSV saturation jump marks the card border
      - Strategy B (coloured sleeve): brightness jump to card's white border
      - Crop to the inner card region and resize back to TARGET dimensions

    Returns: (card_bgr, sleeve_detected, sleeve_width_px)
    """
    h, w   = corrected.shape[:2]
    gray   = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    margin = int(min(h, w) * SLEEVE_OUTER_PCT)

    strips = [
        gray[0:margin, margin:w-margin],       # top
        gray[h-margin:h, margin:w-margin],     # bottom
        gray[margin:h-margin, 0:margin],       # left
        gray[margin:h-margin, w-margin:w],     # right
    ]
    strip_means = [float(np.mean(s)) for s in strips]
    strip_stds  = [float(np.std(s))  for s in strips]

    is_clear_sleeve = all(
        m > SLEEVE_MEAN_MIN and sd < SLEEVE_STD_MAX
        for m, sd in zip(strip_means, strip_stds)
    )
    # Coloured/opaque sleeve: all 4 strips uniformly coloured and same hue as each other
    mean_spread       = max(strip_means) - min(strip_means)
    is_coloured_sleeve = (
        not is_clear_sleeve
        and all(sd < SLEEVE_STD_MAX for sd in strip_stds)
        and mean_spread < SLEEVE_COLOUR_SPREAD_MAX
    )

    is_sleeve = is_clear_sleeve or is_coloured_sleeve
    if not is_sleeve:
        return corrected, False, 0

    # Find where each edge transitions from sleeve to card.
    # Clear sleeve: saturation jump (coloured card border visible against pale sleeve).
    # Coloured sleeve: brightness jump (card's white border brighter than dark sleeve).
    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)

    sleeve_mean_brightness = float(np.mean(strip_means))

    def _find_inset_row(start: int, stop: int, step: int) -> int:
        for y in range(start, stop, step):
            row_sat   = float(np.mean(sat[y, margin:w-margin]))
            row_gray  = float(np.mean(gray[y, margin:w-margin]))
            if is_clear_sleeve and row_sat > 30:
                return y
            if is_coloured_sleeve and row_gray > sleeve_mean_brightness + SLEEVE_COLOUR_JUMP:
                return y
        return start

    def _find_inset_col(start: int, stop: int, step: int) -> int:
        for x in range(start, stop, step):
            col_sat   = float(np.mean(sat[margin:h-margin, x]))
            col_gray  = float(np.mean(gray[margin:h-margin, x]))
            if is_clear_sleeve and col_sat > 30:
                return x
            if is_coloured_sleeve and col_gray > sleeve_mean_brightness + SLEEVE_COLOUR_JUMP:
                return x
        return start

    top_inset    = _find_inset_row(margin,     h // 2,      1)
    bottom_inset = _find_inset_row(h - margin, h // 2,     -1)
    left_inset   = _find_inset_col(margin,     w // 2,      1)
    right_inset  = _find_inset_col(w - margin, w // 2,     -1)

    sleeve_width = min(top_inset, left_inset)
    cropped      = corrected[top_inset:bottom_inset, left_inset:right_inset]
    if cropped.size == 0:
        return corrected, False, 0

    resized = cv2.resize(cropped, (TARGET_CARD_W, TARGET_CARD_H), interpolation=cv2.INTER_LANCZOS4)
    return resized, True, sleeve_width


# ══════════════════════════════════════════════════════════════════════════════
# CENTERING MEASUREMENT
# Based on NickPiscitelli/pokemon-card-analyzer border-width approach
# ══════════════════════════════════════════════════════════════════════════════
def _centering_border_px(gray_line: np.ndarray, sat_line: np.ndarray, reverse: bool) -> float:
    """
    Two-phase border width measurement along one scan line.

    Phase 1 — skip scanner background (LiDE 300: ~232-238 gray, very uniform).
    A 5-pixel sliding window with mean in [228, 242] and std < 8 is scanner bg;
    scanning stops as soon as a non-bg window is found.  For photo images the
    background is dark (< 228) so Phase 1 ends immediately at the image edge.

    Phase 2 — from the card edge found in Phase 1, scan inward until saturation
    exceeds CENTERING_BORDER_SAT_MAX.  The silver/white printed border has S≈0-20;
    the coloured card frame has S>50.  The distance from the card edge to the first
    coloured pixel is the printed border width on that side.
    """
    WIN = 5
    LO, HI, MAX_STD = 228.0, 242.0, 8.0

    if reverse:
        gray_line = gray_line[::-1]
        sat_line  = sat_line[::-1]

    n = len(gray_line)

    # Phase 1: find where scanner background ends
    card_edge = 0
    for i in range(n - WIN + 1):
        win = gray_line[i : i + WIN]
        if not (LO <= float(np.mean(win)) <= HI and float(np.std(win)) < MAX_STD):
            card_edge = i
            break

    # Phase 2: measure from card edge to first coloured content.
    # Require CONSEC consecutive pixels above threshold to avoid stopping inside
    # iridescent metallic (CRI) borders which have occasional high-sat spikes.
    thr    = CENTERING_BORDER_SAT_MAX
    CONSEC = 5
    run    = 0
    for i in range(card_edge, n):
        if sat_line[i] > thr:
            run += 1
            if run >= CONSEC:
                return float(i - CONSEC - card_edge + 1)
        else:
            run = 0

    return 0.0


def measure_centering(corrected: np.ndarray,
                      scanner_bg: tuple = None) -> CenteringResult:
    """
    Measure outer printed border widths on all 4 sides.

    Uses _centering_border_px which first skips scanner background then
    measures to the first coloured content (saturation > CENTERING_BORDER_SAT_MAX).
    Called on the full corrected image (including any scanner bg).

    scanner_bg: (left, right, top, bot) pixel counts of scanner background on each
    side, from _detect_scanner_bg_extents.  When a side has 0 bg AND its measured
    border is < 5 px the card is near the scan edge and that measurement is
    unreliable — substitute the opposite side's value and log a warning.
    """
    h, w   = corrected.shape[:2]
    gray   = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv    = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    sat    = hsv[:, :, 1].astype(np.float32)

    left_vals, right_vals, top_vals, bottom_vals = [], [], [], []
    for row in np.linspace(int(h*0.15), int(h*0.85), CENTERING_SCAN_SAMPLES, dtype=int):
        left_vals.append(_centering_border_px(gray[row, :],  sat[row, :],  False))
        right_vals.append(_centering_border_px(gray[row, :], sat[row, :],  True))

    for col in np.linspace(int(w*0.15), int(w*0.85), CENTERING_SCAN_SAMPLES, dtype=int):
        top_vals.append(_centering_border_px(gray[:, col],   sat[:, col],  False))
        bottom_vals.append(_centering_border_px(gray[:, col], sat[:, col], True))

    L = max(float(np.median(left_vals)),  1)
    R = max(float(np.median(right_vals)), 1)
    T = max(float(np.median(top_vals)),   1)
    B = max(float(np.median(bottom_vals)),1)

    # Clipped-side correction: if scanner bg is 0 on a side and the measured
    # border is < 5 px, the card is too close to the scan edge and that side's
    # border is not visible.  Assume symmetric (use the opposite side's value).
    MIN_READABLE = 5.0
    if scanner_bg is not None:
        sl, sr, st, sb = scanner_bg
        if sr == 0 and R < MIN_READABLE:
            logging.warning(f"centering: right side near scan edge (R border={R:.0f}px) — assuming symmetric L/R")
            R = L
        elif sl == 0 and L < MIN_READABLE:
            logging.warning(f"centering: left side near scan edge (L border={L:.0f}px) — assuming symmetric L/R")
            L = R
        if st == 0 and T < MIN_READABLE:
            logging.warning(f"centering: top side near scan edge (T border={T:.0f}px) — assuming symmetric T/B")
            T = B
        elif sb == 0 and B < MIN_READABLE:
            logging.warning(f"centering: bottom side near scan edge (B border={B:.0f}px) — assuming symmetric T/B")
            B = T
    lr, tb = L+R, T+B
    lp, rp = L/lr*100, R/lr*100
    tp, bp = T/tb*100, B/tb*100

    worse_lr = max(lp, rp)
    worse_tb = max(tp, bp)
    logging.info(f"measure_centering: L={lp:.1f}% R={rp:.1f}% T={tp:.1f}% B={bp:.1f}% (px L={L:.0f} R={R:.0f} T={T:.0f} B={B:.0f})")
    c_grade  = 1  # below PSA 7 default
    for g in sorted(CENTERING_GRADE_THRESHOLDS):  # 7 → 8 → 9 → 10
        if worse_lr <= CENTERING_GRADE_THRESHOLDS[g] and worse_tb <= CENTERING_GRADE_THRESHOLDS[g]:
            c_grade = g
        else:
            break

    return CenteringResult(
        left_pct        = round(lp, 1),
        right_pct       = round(rp, 1),
        top_pct         = round(tp, 1),
        bottom_pct      = round(bp, 1),
        centering_grade = max(c_grade, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CORNER ANALYSIS
# Combines whitening % (visual wear) + Shi-Tomasi sharpness (corner tip integrity)
# Approach from: ScienceDirect automated corner grading paper +
#                rthorst/mint_condition (low-level visual feature importance)
# ══════════════════════════════════════════════════════════════════════════════
def _corner_whitening(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    ring = np.concatenate([gray[0,:], gray[-1,:], gray[1:-1,0], gray[1:-1,-1]])
    bg   = float(np.median(ring))
    return round(np.sum(gray > bg + CORNER_WHITENING_DELTA) / gray.size * 100, 1)


def _corner_sharpness(crop: np.ndarray) -> float:
    """
    Shi-Tomasi corner response at the expected corner tip location.
    Higher response = sharper corner (less worn). Scale: 0–1 (normalised).
    Approach from ScienceDirect automated corner grading paper.
    """
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (CORNER_SHARPNESS_BLUR_K, CORNER_SHARPNESS_BLUR_K), 0)
    corners = cv2.goodFeaturesToTrack(
        blur,
        maxCorners    = 1,
        qualityLevel  = CORNER_SHARPNESS_QUALITY,
        minDistance   = CORNER_SHARPNESS_MIN_DIST,
    )
    if corners is None or len(corners) == 0:
        return 0.0
    # Eigenvalue-based response: how corner-like is the tip?
    # Compute Harris response at detected point
    dst = cv2.cornerHarris(blur, blockSize=3, ksize=3, k=0.04)
    cx, cy = int(corners[0][0][0]), int(corners[0][0][1])
    cx = max(1, min(cx, dst.shape[1]-2))
    cy = max(1, min(cy, dst.shape[0]-2))
    response = float(dst[cy, cx])
    # Normalise against the max response in the image
    max_r    = float(dst.max()) if dst.max() > 0 else 1.0
    return round(response / max_r, 3)


def analyze_corners(corrected: np.ndarray) -> CornerResult:
    h, w = corrected.shape[:2]
    s    = CORNER_CROP_PX
    # TL, TR, BR, BL crops
    crops = [
        corrected[0:s,   0:s  ],
        corrected[0:s,   w-s:w],
        corrected[h-s:h, w-s:w],
        corrected[h-s:h, 0:s  ],
    ]
    whites    = [_corner_whitening(c) for c in crops]
    sharpness = [_corner_sharpness(c) for c in crops]

    worst   = max(whites)
    logging.info(f"analyze_corners: TL={whites[0]:.1f}% TR={whites[1]:.1f}% BR={whites[2]:.1f}% BL={whites[3]:.1f}% worst={worst:.1f}%")
    c_grade = 1  # below PSA 7 default
    for g in sorted(CORNER_GRADE_THRESHOLDS):  # 7 → 8 → 9 → 10
        if worst <= CORNER_GRADE_THRESHOLDS[g]:
            c_grade = g
        else:
            break

    return CornerResult(
        tl_whitening = whites[0],    tr_whitening = whites[1],
        br_whitening = whites[2],    bl_whitening = whites[3],
        tl_sharpness = sharpness[0], tr_sharpness = sharpness[1],
        br_sharpness = sharpness[2], bl_sharpness = sharpness[3],
        corner_grade = max(c_grade, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# HOLO / FOIL DETECTION
# Multi-signal approach: saturation variance + brightness variance + sparkle pixels
# ══════════════════════════════════════════════════════════════════════════════
def detect_holo(corrected: np.ndarray) -> HoloResult:
    """
    Classify whether the card has holographic/foil treatment.

    Three independent signals — card is flagged holo when ≥ 2 fire:
      1. sat_grad    — mean Sobel magnitude on the HSV-S channel.
                       Holo foil has a fine dot/stripe pattern → rapid saturation changes.
                       Smooth artwork colour washes (non-holo) have low gradient.
                       Most reliable single discriminator across all holo types.
      2. local_sat_std — mean saturation std computed inside 16×16 patches.
                       Holo has internal texture within each patch; non-holo artwork
                       uses smooth gradients → low within-patch variance.
      3. sparkle_pct — % artwork pixels with V>235 (very bright).
                       Foil glint under any overhead illumination.

    Thresholds calibrated against Bulbapedia reference card images (cosmos, tinsel,
    starlight, reverse holo patterns) versus a confirmed non-holo card.

    Region: artwork area rows 8–60%, cols 5–95% of the corrected card image.
    Excludes outer white border, name bar, and text box.
    """
    h, w = corrected.shape[:2]
    art  = corrected[int(h*0.08):int(h*0.60), int(w*0.05):int(w*0.95)]
    hsv  = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1].astype(float)
    val  = hsv[:, :, 2].astype(float)

    # Signal 1: saturation gradient (Sobel on S channel)
    sat8 = sat.astype(np.uint8)
    sx   = cv2.Sobel(sat8, cv2.CV_64F, 1, 0, ksize=3)
    sy   = cv2.Sobel(sat8, cv2.CV_64F, 0, 1, ksize=3)
    sat_grad = float(np.mean(np.sqrt(sx**2 + sy**2)))

    # Signal 2: local saturation std (16×16 patches)
    ph = 16
    patch_stds = [
        float(np.std(sat[y:y+ph, x:x+ph]))
        for y in range(0, sat.shape[0] - ph, ph)
        for x in range(0, sat.shape[1] - ph, ph)
    ]
    local_sat_std = float(np.mean(patch_stds)) if patch_stds else 0.0

    # Signal 3: sparkle pixels (overhead foil glint)
    sparkle_pct = float(np.sum(val > 235)) / max(val.size, 1) * 100.0

    signals = [
        sat_grad      > HOLO_SAT_GRAD_MIN,
        local_sat_std > HOLO_LOCAL_SAT_STD_MIN,
        sparkle_pct   > HOLO_SPARKLE_PCT_MIN,
    ]
    score   = sum(signals)
    is_holo = score >= 2
    conf    = {3: "high", 2: "medium", 1: "low", 0: "none"}[score]

    logging.info(
        f"detect_holo: sat_grad={sat_grad:.1f}  local_sat_std={local_sat_std:.1f}  "
        f"sparkle={sparkle_pct:.2f}%  score={score}  holo={is_holo}"
    )
    return HoloResult(
        is_holo       = is_holo,
        confidence    = conf,
        sat_grad      = round(sat_grad, 1),
        local_sat_std = round(local_sat_std, 1),
        sparkle_pct   = round(sparkle_pct, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRINT LINE / SURFACE DEFECT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_print_lines(corrected: np.ndarray) -> PrintLineResult:
    """
    Sobel + HoughLinesP on art area to find consistent linear print artifacts.
    Only nearly-horizontal lines are counted (±10°): real print defects run parallel
    to the print direction; artwork edges can be at any angle and are excluded.
    Surface scratches require raking-light photography — always flagged unverified.
    """
    h, w   = corrected.shape[:2]
    art    = corrected[int(h*0.10):int(h*0.60), int(w*0.05):int(w*0.95)]
    gray   = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    sx     = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy     = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag    = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))
    lines  = cv2.HoughLinesP(
        mag, rho=1, theta=np.pi/180,
        threshold=PRINT_LINE_HOUGH_THRESHOLD,
        minLineLength=PRINT_LINE_MIN_LENGTH,
        maxLineGap=PRINT_LINE_MAX_GAP,
    )
    coords = []
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = (int(v) for v in ln.flatten())
            dx, dy = x2 - x1, y2 - y1
            # Keep only nearly-horizontal lines (angle within ±10° of horizontal)
            if dx == 0:
                continue
            angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
            if angle_deg <= 10 or angle_deg >= 170:
                coords.append((x1, y1, x2, y2))
    return PrintLineResult(
        line_count = len(coords),
        flagged    = len(coords) >= PRINT_LINE_FLAG_COUNT,
        coords     = coords,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GRADE ASSESSMENT + REMEDIATION
# 4-axis framework from crimsonthinker/psa_pokemon_cards
# ══════════════════════════════════════════════════════════════════════════════
def grade_card(centering:   CenteringResult,
               corners:     CornerResult,
               print_lines: PrintLineResult,
               holo:        HoloResult) -> GradeAssessment:
    """
    Assign a probable PSA grade band.
    Measurable axes: centering + corners.  Surface always unverified without raking light.
    """
    remediation: List[str] = []

    # ── Centering ─────────────────────────────────────────────────────────
    cg = centering.centering_grade
    if cg < 10:
        _min_g     = min(CENTERING_GRADE_THRESHOLDS)
        next_g     = cg + 1 if (cg + 1) in CENTERING_GRADE_THRESHOLDS else _min_g
        nt         = CENTERING_GRADE_THRESHOLDS[next_g]
        target_str = f"first target PSA {next_g} requires ≤{nt:.0f}/≥{100-nt:.0f}"
        remediation.append(
            f"CENTERING [PSA {cg}]: {centering.lr_ratio} L/R, {centering.tb_ratio} T/B "
            f"({target_str}). "
            f"Confirm with ruler on physical card — scan perspective can skew measurements."
        )
    else:
        remediation.append(
            f"CENTERING [PSA 10]: {centering.lr_ratio} L/R, {centering.tb_ratio} T/B — "
            f"within PSA 10 threshold. Confirm with ruler on physical card."
        )

    # ── Corners ───────────────────────────────────────────────────────────
    co            = corners.corner_grade
    worst_w       = corners.worst_whitening()
    worst_s       = min(corners.sharpness_scores())
    corner_labels = ["TL", "TR", "BR", "BL"]
    worst_label   = corner_labels[corners.whitening_scores().index(worst_w)]
    severity      = "significant" if worst_w > 20 else "moderate" if worst_w > 12 else "minor"
    if co < 10:
        _co_next  = co + 1 if (co + 1) in CORNER_GRADE_THRESHOLDS else min(CORNER_GRADE_THRESHOLDS)
        _co_th    = CORNER_GRADE_THRESHOLDS[_co_next]
        remediation.append(
            f"CORNERS [PSA {co}]: {severity} whitening on {worst_label} ({worst_w:.0f}% — "
            f"Shi-Tomasi tip sharpness min {worst_s:.3f}). "
            f"Inspect all 4 corners under 10x loupe for chipping, fraying, or micro-tears. "
            f"First target PSA {_co_next} requires worst corner ≤{_co_th}%."
        )
    else:
        remediation.append(
            f"CORNERS [PSA 10]: whitening clean (worst {worst_w:.0f}% on {worst_label}, "
            f"threshold ≤{CORNER_GRADE_THRESHOLDS[10]}%). "
            f"Still inspect under 10x loupe — algorithm cannot detect micro-chipping."
        )

    # ── Holo note ─────────────────────────────────────────────────────────
    if holo.is_holo:
        remediation.append(
            f"HOLO/FOIL [{holo.confidence} confidence]: holographic/foil treatment detected "
            f"(sat_grad={holo.sat_grad}, local_sat_std={holo.local_sat_std}, "
            f"sparkle={holo.sparkle_pct:.1f}%). "
            f"Foil cards command higher premiums — worth grading if centering and corners pass. "
            f"Surface MUST be assessed under angled/raking light (45°) to separate shimmer "
            f"from actual scratches or foil layer damage."
        )

    # ── Print lines / surface defects ─────────────────────────────────────
    # Suppress print-line flag for confirmed holo cards — foil embossing creates
    # many false horizontal lines in Sobel/Hough on an overhead scan.
    effective_flagged = print_lines.flagged and not holo.is_holo
    if effective_flagged:
        remediation.append(
            f"SURFACE [unverified + artifacts]: {print_lines.line_count} linear artifacts "
            f"detected under flat light. Re-photograph at 45° raking angle — if lines "
            f"persist they are print defects (reduces surface grade). "
            f"Surface condition cannot be fully assessed without raking light."
        )
    elif holo.is_holo and print_lines.flagged:
        remediation.append(
            f"SURFACE [unverified — holo]: {print_lines.line_count} lines detected but "
            f"suppressed (foil embossing creates false positives in overhead scan). "
            f"Photograph at 45° raking angle to distinguish foil texture from actual defects."
        )
    else:
        remediation.append(
            "SURFACE [unverified]: no linear artifacts under flat light. "
            "Re-photograph under raking/angled light (45° off-axis) to reveal "
            "scratches, print defects, and surface wear. Required before grading."
        )


    # ── Overall grade ──────────────────────────────────────────────────────
    measurable = min(cg, co)
    limiting_factor = " + ".join(
        name for grade, name in [(cg, "centering"), (co, "corners")]
        if grade == measurable
    )

    if measurable == 10:
        band   = "PSA 10 candidate (surface and back unverified)"
        factor = "none detected — surface (raking light) and back still need verification"
    elif measurable == 9:
        band   = f"PSA 9 candidate (limited by {limiting_factor})"
        factor = limiting_factor
    elif measurable == 8:
        band   = f"PSA 8 candidate (limited by {limiting_factor})"
        factor = limiting_factor
    elif measurable == 7:
        band   = f"PSA 7 candidate (limited by {limiting_factor})"
        factor = limiting_factor
    else:
        band   = f"PSA {measurable} or below (limited by {limiting_factor})"
        factor = limiting_factor

    return GradeAssessment(
        centering_grade  = cg,
        corner_grade     = co,
        surface_grade    = "unverified",
        probable_grade   = measurable,
        grade_band       = band,
        limiting_factor  = factor,
        surface_verified = False,
        is_holo          = holo.is_holo,
        remediation      = remediation,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OCR — card name, number, HP
# ══════════════════════════════════════════════════════════════════════════════
def _crop_region(img: np.ndarray, region: Tuple) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region
    return img[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]


def _upscale_colour(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * OCR_SCALE, h * OCR_SCALE), interpolation=cv2.INTER_LANCZOS4)


def _binarize_for_tesseract(crop: np.ndarray) -> np.ndarray:
    big  = _upscale_colour(crop)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def _ocr_region(img: np.ndarray, region: Tuple, numeric: bool = False) -> str:
    if OCR_ENGINE is None:
        return ""
    crop = _crop_region(img, region)
    engine, obj = OCR_ENGINE
    if engine == "easyocr":
        colour_big = _upscale_colour(crop)
        results = obj.readtext(colour_big, detail=0)
        logging.debug(f"easyocr raw results: {results}")
        return " ".join(results).strip()
    import pytesseract
    proc = _binarize_for_tesseract(crop)
    cfg = "--psm 7 --oem 3"
    cfg += (" -c tessedit_char_whitelist=0123456789/" if numeric
            else " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'")
    return pytesseract.image_to_string(proc, config=cfg).strip()


_CARD_NOISE_WORDS = {
    "me", "hp", "lv", "ex", "gx", "vmax", "vstar", "basic",
    "stage", "item", "trainer", "the", "a", "an", "in", "of",
}

def _clean_ocr_name(raw: str) -> str:
    # Extract only Latin-script words (handles Japanese cards where OCR returns mix)
    raw = re.sub(r"[^A-Za-z\- ']", " ", raw).strip()
    words = raw.split()
    # Drop leading fragments: words of 1-2 chars or known noise words
    while words and (len(words[0]) <= 2 or words[0].lower() in _CARD_NOISE_WORDS):
        words = words[1:]
    # Keep only words that look like proper names (≥3 chars, mostly alphabetic)
    words = [w for w in words if len(w) >= 3 and sum(c.isalpha() for c in w) >= 2]
    return " ".join(words)


def ocr_card(corrected: np.ndarray) -> Tuple[str, str, str]:
    """Returns (card_name, card_number, hp). Always attempted."""
    raw_name = _ocr_region(corrected, OCR_NAME_REGION)
    name     = _clean_ocr_name(raw_name)
    logging.debug(f"OCR name raw='{raw_name}'  cleaned='{name}'")
    num    = _ocr_region(corrected, OCR_NUM_REGION, numeric=True)
    hp_raw = _ocr_region(corrected, OCR_HP_REGION,  numeric=True)
    m      = re.search(r"\d{2,3}/\d{2,3}", num)
    number = m.group(0) if m else re.sub(r"[^\d/]", "", num).strip()
    hp_m   = re.search(r"\d+", hp_raw)
    hp     = hp_m.group(0) if hp_m else ""
    return name, number, hp


def _detect_scanner_bg_extents(img: np.ndarray):
    """
    Detect how many pixels of scanner background (LiDE 300: ~232-238 gray,
    very uniform) sit on each side of a corrected scanner image.

    Returns (left, right, top, bottom) pixel counts.  Uses a 5-pixel sliding
    window from each edge: mean in [228, 242] AND std < 8 → scanner background.
    Sampling 12 rows/cols across the central 60 % of each edge and taking the
    median makes the estimate robust against isolated bright card elements
    (name plate, stage badge) near the edge of the window.
    """
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    WIN   = 5
    LO, HI, MAX_STD = 228.0, 242.0, 8.0

    def edge_bg(line: np.ndarray) -> int:
        n = len(line)
        for i in range(n - WIN + 1):
            win = line[i : i + WIN]
            if not (LO <= float(np.mean(win)) <= HI and float(np.std(win)) < MAX_STD):
                return max(0, i)
        return n

    step_r = max(1, int(h * 0.05))
    step_c = max(1, int(w * 0.05))
    rows   = range(int(h * 0.2), int(h * 0.8), step_r)
    cols   = range(int(w * 0.2), int(w * 0.8), step_c)

    left  = int(np.median([edge_bg(gray[r, :])    for r in rows]))
    right = int(np.median([edge_bg(gray[r, ::-1]) for r in rows]))
    top   = int(np.median([edge_bg(gray[:, c])    for c in cols]))
    bot   = int(np.median([edge_bg(gray[::-1, c]) for c in cols]))
    return left, right, top, bot


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CARD PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def _is_card_back(corrected: np.ndarray) -> bool:
    """
    Detect if the corrected image shows the card BACK rather than the front.
    The Pokemon card back is dark red/blue (Pokeball design) with almost no
    white pixels.  The front always has a white border + text box.
    Heuristic: if <1% of pixels exceed brightness 200 AND mean saturation > 60,
    it's almost certainly the back.
    """
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    white_pct = float(np.mean(gray > 200)) * 100
    sat_mean  = float(np.mean(hsv[:, :, 1]))
    return white_pct < 1.0 and sat_mean > 60


def process_card_image(bgr: np.ndarray, label: str, out_dir: Path) -> CardResult:
    result = CardResult(filename=label)
    try:
        # 1. Perspective correct outer boundary (sleeve or card)
        corrected, _, is_scan = perspective_correct(bgr)
        corrected = auto_orient(corrected, scanner=is_scan)

        # Guard: detect card placed face-up (back visible)
        if _is_card_back(corrected):
            raise ValueError(
                "Card back detected — place card FACE DOWN (art side against scanner glass) and rescan."
            )

        # 2. Build grading_img: card only, no scanner background.
        #    For scanner images, use the projection bbox on the *original* scan to
        #    crop exactly to the card boundary — no bg-inset guessing.  This is the
        #    same landmark-anchor approach as face detection: find the distinctive
        #    boundary, crop to known dimensions, grade that clean image.
        h_c, w_c = corrected.shape[:2]
        if is_scan:
            bbox = _find_card_bbox_projection(bgr)
            if bbox is not None:
                rx1, ry1, rx2, ry2 = bbox
                raw_card = bgr[ry1:ry2, rx1:rx2]
                grading_img = cv2.resize(
                    raw_card,
                    (TARGET_CARD_W, TARGET_CARD_H),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                grading_img = auto_orient(grading_img, scanner=True)
                left_bg = right_bg = top_bg = bot_bg = 0
                logging.info(f"grading_img: projection bbox {rx2-rx1}×{ry2-ry1} → direct crop")
            else:
                # Fallback: scanner bg inset on corrected image
                left_bg, right_bg, top_bg, bot_bg = _detect_scanner_bg_extents(corrected)
                logging.info(f"scanner bg inset: L={left_bg} R={right_bg} T={top_bg} B={bot_bg}")
                cap = min(h_c, w_c) // 4
                x1 = min(left_bg, cap);  x2 = w_c - min(right_bg, cap)
                y1 = min(top_bg,  cap);  y2 = h_c - min(bot_bg,   cap)
                grading_img = cv2.resize(
                    corrected[y1:y2, x1:x2],
                    (TARGET_CARD_W, TARGET_CARD_H),
                    interpolation=cv2.INTER_LANCZOS4,
                ) if x2 > x1 + 20 and y2 > y1 + 20 else corrected
        else:
            grading_img = corrected
            left_bg = right_bg = top_bg = bot_bg = 0

        _sbg = (left_bg, right_bg, top_bg, bot_bg) if is_scan else None
        result.centering   = measure_centering(corrected, scanner_bg=_sbg)
        result.corners     = analyze_corners(grading_img)
        result.holo        = detect_holo(grading_img)
        result.print_lines = detect_print_lines(grading_img)

        # 3. Grade every card
        result.grade = grade_card(
            result.centering, result.corners,
            result.print_lines, result.holo,
        )

        # 5. Save compressed display image — 630×880 JPEG (~150 KB, good enough for GUI)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        img_small = cv2.resize(corrected, (630, 880), interpolation=cv2.INTER_AREA)
        out_path  = IMAGES_DIR / f"{label}.jpg"
        cv2.imwrite(str(out_path), img_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result.corrected_img_path = str(out_path)

        # 6. OCR — always attempted
        name, number, hp = ocr_card(grading_img)
        result.identification = IDResult(
            card_name      = name,
            card_number    = number,
            hp             = hp,
            ocr_raw_name   = name,
            ocr_raw_number = number,
            ocr_confidence = "medium" if name else "none",
        )

        if not name and result.grade:
            result.grade.remediation.append(
                "IDENTIFICATION: OCR could not read card name. "
                "Install easyocr (pip install easyocr) and re-run, or identify manually."
            )

    except Exception as e:
        result.error = str(e)
        logging.error(f"{label}: {e}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BINDER GRID EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_binder_cards(page_bgr: np.ndarray) -> List[np.ndarray]:
    h, w   = page_bgr.shape[:2]
    mx, my = int(w*BINDER_MARGIN_PCT), int(h*BINDER_MARGIN_PCT)
    uw, uh = w-2*mx, h-2*my
    cw, ch = uw//BINDER_COLS, uh//BINDER_ROWS
    px, py = int(cw*BINDER_CELL_PAD_PCT), int(ch*BINDER_CELL_PAD_PCT)
    cells  = []
    for row in range(BINDER_ROWS):
        for col in range(BINDER_COLS):
            x1 = mx + col*cw + px
            y1 = my + row*ch + py
            cells.append(page_bgr[y1:y1+ch-2*py, x1:x1+cw-2*px])
    return cells


def _is_blank_cell(bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) > 230 and float(np.std(gray)) < 12


# ══════════════════════════════════════════════════════════════════════════════
# PDF CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
def pdf_to_pages(pdf_path: Path) -> List[np.ndarray]:
    if not PDF_OK:
        raise RuntimeError("pypdfium2 not installed: pip install pypdfium2")
    doc   = pdfium.PdfDocument(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        bm  = doc[i].render(scale=PDF_RENDER_SCALE)
        arr = np.array(bm.to_pil().convert("RGB"))
        pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def process_image_folder(folder: Path, out_dir: Path) -> List[CardResult]:
    files  = sorted(f for f in folder.iterdir() if f.suffix.lower() in _IMG_EXT)
    fronts = {f.stem.replace("_front","").replace("_back",""): f for f in files if "_front" in f.stem}
    backs  = {f.stem.replace("_front","").replace("_back",""): f for f in files if "_back"  in f.stem}
    singles= [f for f in files if "_front" not in f.stem and "_back" not in f.stem]
    keys   = list(fronts) + [f.stem for f in singles]
    results= []
    for i, key in enumerate(keys, 1):
        front = fronts.get(key) or next((f for f in singles if f.stem == key), None)
        back  = backs.get(key)
        if not front: continue
        logging.info(f"[{i}/{len(keys)}] {front.name}")
        result = process_card_image(cv2.imread(str(front)), f"card_{i:04d}_{key}", out_dir)
        result.filename   = front.name
        result.card_index = i
        results.append(result)
    return results


def process_binder_pdf(pdf_path: Path, out_dir: Path, start_n: int = 1) -> List[CardResult]:
    pages   = pdf_to_pages(pdf_path)
    results = []
    card_n  = start_n
    for pg_i, page in enumerate(pages, 1):
        for cell_j, cell in enumerate(extract_binder_cards(page), 1):
            if _is_blank_cell(cell): continue
            label  = f"p{pg_i:02d}_c{cell_j:02d}"
            logging.info(f"[{card_n}] {pdf_path.name} pg{pg_i} cell{cell_j}")
            result = process_card_image(cell, label, out_dir)
            result.filename   = f"{pdf_path.name}::page{pg_i}:cell{cell_j}"
            result.card_index = card_n
            results.append(result)
            card_n += 1
    return results


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
CSV_FIELDS = [
    "card_index","filename",
    # Centering
    "centering_lr","centering_tb","centering_grade",
    # Corners (whitening + sharpness)
    "corner_tl_white","corner_tr_white","corner_br_white","corner_bl_white",
    "corner_tl_sharp","corner_tr_sharp","corner_br_sharp","corner_bl_sharp","corner_grade",
    # Holo detection
    "is_holo","holo_confidence","holo_sat_grad","holo_local_sat_std","holo_sparkle_pct",
    # Print lines
    "print_line_count","print_line_flagged",
    # Grade
    "centering_subgrade","corner_subgrade","surface_grade",
    "probable_grade","grade_band","limiting_factor","surface_verified",
    "remediation",
    # Identification (OCR only — no network)
    "card_name","card_number","hp",
    "ocr_raw_name","ocr_raw_number","ocr_confidence",
    "corrected_img_path","error",
]


def _row(r: CardResult) -> dict:
    c   = r.centering      or CenteringResult()
    co  = r.corners        or CornerResult()
    ho  = r.holo           or HoloResult()
    pl  = r.print_lines    or PrintLineResult()
    gr  = r.grade          or GradeAssessment()
    idr = r.identification or IDResult()
    return dict(
        card_index          = r.card_index,
        filename            = r.filename,
        centering_lr        = c.lr_ratio,
        centering_tb        = c.tb_ratio,
        centering_grade     = c.centering_grade,
        corner_tl_white     = co.tl_whitening,
        corner_tr_white     = co.tr_whitening,
        corner_br_white     = co.br_whitening,
        corner_bl_white     = co.bl_whitening,
        corner_tl_sharp     = co.tl_sharpness,
        corner_tr_sharp     = co.tr_sharpness,
        corner_br_sharp     = co.br_sharpness,
        corner_bl_sharp     = co.bl_sharpness,
        corner_grade        = co.corner_grade,
        is_holo             = ho.is_holo,
        holo_confidence     = ho.confidence,
        holo_sat_grad       = ho.sat_grad,
        holo_local_sat_std  = ho.local_sat_std,
        holo_sparkle_pct    = ho.sparkle_pct,
        print_line_count    = pl.line_count,
        print_line_flagged  = pl.flagged,
        centering_subgrade  = gr.centering_grade,
        corner_subgrade     = gr.corner_grade,
        surface_grade       = gr.surface_grade,
        probable_grade      = gr.probable_grade,
        grade_band          = gr.grade_band,
        limiting_factor     = gr.limiting_factor,
        surface_verified    = gr.surface_verified,
        remediation         = " | ".join(gr.remediation),
        card_name           = idr.card_name,
        card_number         = idr.card_number,
        hp                  = idr.hp,
        ocr_raw_name        = idr.ocr_raw_name,
        ocr_raw_number      = idr.ocr_raw_number,
        ocr_confidence      = idr.ocr_confidence,
        corrected_img_path  = r.corrected_img_path,
        error               = r.error,
    )


def _show_portfolio(search: str = "") -> None:
    """Print all cards from the portfolio DB, optionally filtered by name/number."""
    if not DB_PATH.exists():
        print("\n  No portfolio database found. Grade some cards first.")
        return
    con = _open_db()
    SEL = ("SELECT card_name, card_number, hp, scan_index, probable_grade, grade_band, "
           "centering_lr, centering_tb, corner_grade, is_holo, date_scanned FROM portfolio")
    if search:
        rows = con.execute(
            SEL + " WHERE card_name LIKE ? OR card_number LIKE ? "
            "ORDER BY probable_grade DESC, card_name, scan_index",
            (f"%{search}%", f"%{search}%"),
        ).fetchall()
    else:
        rows = con.execute(
            SEL + " ORDER BY probable_grade DESC, card_name, scan_index"
        ).fetchall()
    total = con.execute("SELECT COUNT(*) FROM portfolio").fetchone()[0]
    con.close()

    if not rows:
        print(f"\n  No cards found{' matching ' + repr(search) if search else ''}.")
        return

    hdr = f"  {'Card Name':<28}  {'#':>7}  {'HP':>4}  {'Sc':>2}  {'Gr':>4}  {'C L/R':>9}  {'C T/B':>9}  {'Cor':>3}  {'Holo':>4}  Date"
    sep = f"  {'-'*28}  {'-'*7}  {'-'*4}  {'-'*2}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*3}  {'-'*4}  ----------"
    print(f"\n{'='*102}")
    filter_note = f"  Filter: '{search}'  — " if search else "  "
    print(f"{filter_note}{len(rows)} card{'s' if len(rows)!=1 else ''} / {total} total in portfolio")
    print(hdr)
    print(sep)
    for (name, num, hp, sidx, grade, band, clr, ctb, cor, holo, dt) in rows:
        name  = (name  or "—")[:28]
        num   = (num   or "—")[:7]
        hp    = (hp    or "—")[:4]
        sc    = str(sidx or 1)
        gr    = str(grade) if grade else "—"
        clr   = (clr   or "—")[:9]
        ctb   = (ctb   or "—")[:9]
        cor   = str(cor) if cor is not None else "—"
        hflag = "HOLO" if holo else "    "
        date  = (dt or "")[:10]
        print(f"  {name:<28}  {num:>7}  {hp:>4}  {sc:>2}  {gr:>4}  {clr:>9}  {ctb:>9}  {cor:>3}  {hflag:>4}  {date}")
    print(f"{'='*102}\n")


def _launch_gui() -> None:
    """Tkinter card portfolio viewer."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import Image as PILImage, ImageTk
    except ImportError as e:
        print(f"GUI requires Pillow: pip install pillow  ({e})")
        return

    COL_NAMES = ["id","card_name","card_number","hp","scan_index",
                 "probable_grade","grade_band","centering_lr","centering_tb","centering_grade",
                 "corner_tl","corner_tr","corner_br","corner_bl","corner_grade",
                 "is_holo","print_line","limiting_factor","remediation",
                 "image_path","date_scanned"]

    def _load_cards() -> List[dict]:
        if not DB_PATH.exists():
            return []
        try:
            con = _open_db()
            rows = con.execute("""
                SELECT id, card_name, card_number, hp, scan_index,
                       probable_grade, grade_band, centering_lr, centering_tb, centering_grade,
                       corner_tl, corner_tr, corner_br, corner_bl, corner_grade,
                       is_holo, print_line, limiting_factor, remediation,
                       image_path, date_scanned
                FROM portfolio ORDER BY probable_grade DESC, card_name, scan_index
            """).fetchall()
            con.close()
            return [dict(zip(COL_NAMES, row)) for row in rows]
        except Exception:
            return []

    cards = _load_cards()

    # ── Window ───────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Pokémon Card Portfolio")
    root.configure(bg="#1a1a2e")
    root.geometry("1280x820")
    root.minsize(900, 600)

    DARK  = "#1a1a2e"
    PANEL = "#16213e"
    ACCENT= "#e94560"
    TEXT  = "#eaeaea"
    MUTED = "#8888aa"
    MONO  = ("Consolas", 10)
    TITLE = ("Consolas", 11, "bold")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Card.TFrame",     background=PANEL)
    style.configure("Dark.TLabel",     background=PANEL, foreground=TEXT, font=MONO)
    style.configure("Accent.TLabel",   background=PANEL, foreground=ACCENT, font=TITLE)
    style.configure("Muted.TLabel",    background=PANEL, foreground=MUTED, font=MONO)
    style.configure("Card.TListbox",   background=PANEL, foreground=TEXT, font=MONO)
    style.configure("Thin.TSeparator", background=ACCENT)

    # ── Bottom action bar (packed before side frames so it claims space first) ─
    bar = tk.Frame(root, bg=DARK, pady=6)
    bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8)

    # ── Left panel — card list ───────────────────────────────────────────────
    frame_l = tk.Frame(root, bg=PANEL, width=360)
    frame_l.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0), pady=8)
    frame_l.pack_propagate(False)

    search_var = tk.StringVar()
    search_entry = tk.Entry(frame_l, textvariable=search_var, bg="#0f3460", fg=TEXT,
                            insertbackground=TEXT, font=MONO, relief=tk.FLAT, bd=4)
    search_entry.pack(fill=tk.X, padx=4, pady=(4, 2))
    search_entry.insert(0, "Search…")
    search_entry.bind("<FocusIn>",  lambda e: search_entry.delete(0, tk.END)
                      if search_entry.get() == "Search…" else None)
    search_entry.bind("<FocusOut>", lambda e: search_entry.insert(0, "Search…")
                      if not search_entry.get() else None)

    lb_frame = tk.Frame(frame_l, bg=PANEL)
    lb_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

    scrollbar = tk.Scrollbar(lb_frame, bg=PANEL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(lb_frame, yscrollcommand=scrollbar.set, bg=PANEL, fg=TEXT,
                         selectbackground=ACCENT, selectforeground=TEXT,
                         font=MONO, relief=tk.FLAT, bd=0, activestyle="none",
                         highlightthickness=0)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    count_lbl = tk.Label(frame_l, text="", bg=PANEL, fg=MUTED, font=MONO)
    count_lbl.pack(pady=(0, 4))

    visible_cards: List[dict] = []

    def _populate(query: str = "") -> None:
        listbox.delete(0, tk.END)
        visible_cards.clear()
        q = query.lower().strip()
        for c in cards:
            nm  = (c["card_name"] or "").lower()
            num = (c["card_number"] or "").lower()
            if q and q not in nm and q not in num:
                continue
            idx = c["scan_index"] or 1
            tag = f"  PSA {c['probable_grade'] or '?':>2}  {(c['card_name'] or 'Unknown')[:22]:<22}"
            if idx > 1:
                tag += f"  #{idx}"
            listbox.insert(tk.END, tag)
            visible_cards.append(c)
        count_lbl.config(text=f"{len(visible_cards)} / {len(cards)} cards")

    search_var.trace_add("write", lambda *_: _populate(
        "" if search_var.get() == "Search…" else search_var.get()))
    _populate()

    # ── Right panel — header + image + details ──────────────────────────────
    frame_r = tk.Frame(root, bg=DARK)
    frame_r.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4,8), pady=8)

    # Card header bar
    rh = tk.Frame(frame_r, bg=PANEL, padx=10, pady=8)
    rh.pack(fill=tk.X, pady=(0, 4))
    card_name_var = tk.StringVar(value="")
    tk.Label(rh, textvariable=card_name_var, bg=PANEL, fg=TEXT,
             font=("Consolas", 13, "bold"), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
    del_btn = tk.Button(rh, text="Delete Card", bg="#3a1a2e", fg="#ff5252",
                        font=("Consolas", 9, "bold"), relief=tk.FLAT, padx=10, pady=4,
                        cursor="hand2", activebackground="#5a1a2e", activeforeground="#ff5252",
                        state=tk.DISABLED)
    del_btn.pack(side=tk.RIGHT)

    # ── Card slab (image window + label) ────────────────────────────────────
    SLAB_BDR  = "#111827"   # outer slab body
    SLAB_BG   = "#1a1a2e"   # slab interior
    WIN_BG    = "#07070f"   # clear window behind card
    LABEL_BG  = "#0d1220"   # label area
    LABEL_HDR = "#0f172a"   # label header strip

    slab_outer = tk.Frame(frame_r, bg=SLAB_BDR, padx=6, pady=6)
    slab_outer.pack(fill=tk.X, padx=4, pady=(0, 4))
    slab_inner = tk.Frame(slab_outer, bg=SLAB_BG)
    slab_inner.pack(fill=tk.BOTH)

    # Card window (top of slab — simulates clear plastic over the card)
    card_win = tk.Frame(slab_inner, bg=WIN_BG, pady=10)
    card_win.pack(fill=tk.X)
    img_label = tk.Label(card_win, bg=WIN_BG)
    img_label.pack()

    # Slab label docked at bottom — like a real PSA holder
    tk.Frame(slab_inner, bg=ACCENT, height=2).pack(fill=tk.X)   # accent divider between window and label

    label_area = tk.Frame(slab_inner, bg=LABEL_BG)
    label_area.pack(fill=tk.X)

    # Label header strip: title left, grade box right
    lbl_hdr = tk.Frame(label_area, bg=LABEL_HDR, padx=10, pady=6)
    lbl_hdr.pack(fill=tk.X)
    tk.Label(lbl_hdr, text="P O K É M O N   G R A D E   E S T I M A T E",
             bg=LABEL_HDR, fg=ACCENT, font=("Consolas", 8, "bold")).pack(side=tk.LEFT)
    grade_box_border = tk.Frame(lbl_hdr, bg=ACCENT, padx=2, pady=2)
    grade_box_border.pack(side=tk.RIGHT)
    grade_num_var = tk.StringVar(value="—")
    grade_num_lbl  = tk.Label(grade_box_border, textvariable=grade_num_var,
                               bg=LABEL_HDR, fg=MUTED,
                               font=("Consolas", 28, "bold"), width=3,
                               anchor="center", padx=6, pady=2)
    grade_num_lbl.pack()

    # Band + limiting factor
    band_row = tk.Frame(label_area, bg=LABEL_BG, padx=10, pady=4)
    band_row.pack(fill=tk.X)
    grade_band_var = tk.StringVar(value="")
    tk.Label(band_row, textvariable=grade_band_var, bg=LABEL_BG, fg=TEXT,
             font=("Consolas", 10, "bold"), anchor="w").pack(anchor="w")
    limiting_var = tk.StringVar(value="")
    limiting_lbl = tk.Label(band_row, textvariable=limiting_var,
                             bg=LABEL_BG, fg="#ffab40",
                             font=("Consolas", 9), anchor="w")
    limiting_lbl.pack(anchor="w")

    tk.Frame(label_area, bg="#1e2a40", height=1).pack(fill=tk.X, padx=10)

    # Stats row: CENTERING | CORNERS
    stats_row = tk.Frame(label_area, bg=LABEL_BG, padx=10, pady=6)
    stats_row.pack(fill=tk.X)
    cent_col = tk.Frame(stats_row, bg=LABEL_BG)
    cent_col.pack(side=tk.LEFT, anchor="n", padx=(0, 20))
    tk.Label(cent_col, text="CENTERING", bg=LABEL_BG, fg=ACCENT,
             font=("Consolas", 8, "bold")).pack(anchor="w")
    cent_lr_var = tk.StringVar(); cent_tb_var = tk.StringVar(); cent_g_var = tk.StringVar()
    tk.Label(cent_col, textvariable=cent_lr_var, bg=LABEL_BG, fg=TEXT, font=MONO).pack(anchor="w")
    tk.Label(cent_col, textvariable=cent_tb_var, bg=LABEL_BG, fg=TEXT, font=MONO).pack(anchor="w")
    cent_g_lbl = tk.Label(cent_col, textvariable=cent_g_var, bg=LABEL_BG, font=MONO)
    cent_g_lbl.pack(anchor="w")

    corn_col = tk.Frame(stats_row, bg=LABEL_BG)
    corn_col.pack(side=tk.LEFT, anchor="n")
    tk.Label(corn_col, text="CORNERS", bg=LABEL_BG, fg=ACCENT,
             font=("Consolas", 8, "bold")).pack(anchor="w")
    corn_tl_tr_var = tk.StringVar(); corn_bl_br_var = tk.StringVar(); corn_g_var = tk.StringVar()
    tk.Label(corn_col, textvariable=corn_tl_tr_var, bg=LABEL_BG, fg=TEXT, font=MONO).pack(anchor="w")
    tk.Label(corn_col, textvariable=corn_bl_br_var, bg=LABEL_BG, fg=TEXT, font=MONO).pack(anchor="w")
    corn_g_lbl = tk.Label(corn_col, textvariable=corn_g_var, bg=LABEL_BG, font=MONO)
    corn_g_lbl.pack(anchor="w")

    tk.Frame(label_area, bg="#1e2a40", height=1).pack(fill=tk.X, padx=10)

    # Info row (HP · Holo · Date)
    info_frame = tk.Frame(label_area, bg=LABEL_HDR, padx=10, pady=4)
    info_frame.pack(fill=tk.X)
    info_var = tk.StringVar(value="")
    tk.Label(info_frame, textvariable=info_var, bg=LABEL_HDR, fg=MUTED, font=MONO).pack()

    # Remediation — inside label, only shown when needed
    rem_sep = tk.Frame(label_area, bg=ACCENT, height=1)
    rem_body = tk.Frame(label_area, bg="#160d1a", padx=10, pady=6)
    rem_text = tk.Text(rem_body, bg="#160d1a", fg="#ffab40", font=MONO, relief=tk.FLAT,
                       bd=0, wrap=tk.WORD, highlightthickness=0,
                       state=tk.DISABLED, height=4)
    rem_text.tag_configure("hdr",  foreground=ACCENT, font=("Consolas", 8, "bold"))
    rem_text.tag_configure("item", foreground="#ffab40")

    _photo_ref: List = []
    _selected_card: List[dict] = []

    def _grade_color(g) -> str:
        if not g:  return MUTED
        if g >= 9: return "#00e676"
        if g >= 7: return "#ffab40"
        return "#ff5252"

    def _show(card: dict) -> None:
        _selected_card.clear()
        _selected_card.append(card)
        del_btn.config(state=tk.NORMAL)

        name = card.get("card_name") or "Unknown"
        num  = card.get("card_number") or "—"
        idx  = card.get("scan_index") or 1
        scan_sfx = f"  ·  scan #{idx}" if idx > 1 else ""
        card_name_var.set(f"{name}   #{num}{scan_sfx}")

        # Image
        img_path = card.get("image_path") or ""
        if img_path and Path(img_path).exists():
            try:
                pil = PILImage.open(img_path)
                pil.thumbnail((300, 420), PILImage.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                _photo_ref.clear(); _photo_ref.append(photo)
                img_label.config(image=photo, text="")
            except Exception:
                img_label.config(image="", text="[image error]", fg=MUTED, font=MONO)
        else:
            img_label.config(image="", text="[no image]", fg=MUTED, font=MONO)

        # Grade badge
        g    = card.get("probable_grade") or 0
        band = card.get("grade_band") or "Ungraded"
        lf   = card.get("limiting_factor") or ""
        grade_num_var.set(str(g) if g else "?")
        grade_num_lbl.config(fg=_grade_color(g), bg=LABEL_BG)
        grade_band_var.set(band)
        limiting_var.set(f"Limited by: {lf}" if lf else "")

        # Centering
        clr = card.get("centering_lr") or "—"
        ctb = card.get("centering_tb") or "—"
        cg  = card.get("centering_grade") or 0
        cent_lr_var.set(f"L / R   {clr}")
        cent_tb_var.set(f"T / B   {ctb}")
        cent_g_var.set(f"Grade   PSA {cg}")
        cent_g_lbl.config(fg=_grade_color(cg))

        # Corners
        tl = card.get("corner_tl"); tr = card.get("corner_tr")
        bl = card.get("corner_bl"); br = card.get("corner_br")
        def _cv(v): return f"{v:.1f}%" if v is not None else "—"
        corn_tl_tr_var.set(f"TL {_cv(tl)}   TR {_cv(tr)}")
        corn_bl_br_var.set(f"BL {_cv(bl)}   BR {_cv(br)}")
        cor_g = card.get("corner_grade") or 0
        corn_g_var.set(f"Grade   PSA {cor_g}")
        corn_g_lbl.config(fg=_grade_color(cor_g))

        # Info row
        hp   = card.get("hp") or "—"
        holo = "Holo" if card.get("is_holo") else "Non-holo"
        date = (card.get("date_scanned") or "—")[:10]
        info_var.set(f"{hp} HP   ·   {holo}   ·   {date}")

        # Remediation (inside slab label, shown only when needed)
        rem = (card.get("remediation") or "").strip()
        rem_text.config(state=tk.NORMAL)
        rem_text.delete("1.0", tk.END)
        if rem:
            rem_text.insert(tk.END, "REMEDIATION\n", "hdr")
            for item in rem.split(" | "):
                item = item.strip()
                if item:
                    rem_text.insert(tk.END, f"  ▸ {item}\n", "item")
            rem_sep.pack(fill=tk.X)
            rem_body.pack(fill=tk.X)
            rem_text.pack(fill=tk.X)
        else:
            rem_text.pack_forget()
            rem_body.pack_forget()
            rem_sep.pack_forget()
        rem_text.config(state=tk.DISABLED)

    def _clear_display() -> None:
        card_name_var.set("")
        img_label.config(image="", text="")
        grade_num_var.set("—"); grade_num_lbl.config(fg=MUTED)
        grade_band_var.set(""); limiting_var.set("")
        cent_lr_var.set(""); cent_tb_var.set(""); cent_g_var.set("")
        corn_tl_tr_var.set(""); corn_bl_br_var.set(""); corn_g_var.set("")
        info_var.set("")
        rem_text.pack_forget(); rem_body.pack_forget(); rem_sep.pack_forget()

    def _delete_selected() -> None:
        if not _selected_card:
            return
        card = _selected_card[0]
        name = card.get("card_name") or "Unknown"
        num  = card.get("card_number") or "—"
        if not messagebox.askyesno("Delete Card",
                                   f"Delete '{name}  #{num}' from portfolio?\n\nThis also removes the saved image."):
            return
        img = card.get("image_path") or ""
        if img and Path(img).exists():
            try: Path(img).unlink()
            except Exception: pass
        try:
            con = _open_db()
            con.execute("DELETE FROM portfolio WHERE id=?", (card["id"],))
            con.commit(); con.close()
        except Exception as exc:
            messagebox.showerror("Error", f"Could not delete: {exc}")
            return
        _selected_card.clear()
        del_btn.config(state=tk.DISABLED)
        _clear_display()
        nonlocal cards
        cards = _load_cards()
        _populate()
        n = len(cards)
        status_var.set(f"Deleted. {n} card{'s' if n != 1 else ''} in portfolio.")

    del_btn.config(command=_delete_selected)

    def _on_select(event=None) -> None:
        sel = listbox.curselection()
        if not sel or sel[0] >= len(visible_cards):
            return
        _show(visible_cards[sel[0]])

    listbox.bind("<<ListboxSelect>>", _on_select)
    if visible_cards:
        listbox.selection_set(0)
        _show(visible_cards[0])

    # ── Action bar buttons ───────────────────────────────────────────────────
    import threading as _th

    btn_style = {
        "bg": ACCENT, "fg": TEXT, "font": ("Consolas", 10, "bold"),
        "relief": tk.FLAT, "padx": 16, "pady": 6, "cursor": "hand2",
        "activebackground": "#c73652", "activeforeground": TEXT,
    }
    status_var = tk.StringVar(value="Scan a card, then click Grade Report to save it.")
    tk.Label(bar, textvariable=status_var, bg=DARK, fg=MUTED, font=MONO).pack(
        side=tk.RIGHT, padx=8)

    if getattr(sys, "frozen", False):
        _exe_cmd: List[str] = [sys.executable]
    else:
        _exe_cmd = [sys.executable, str(Path(__file__).resolve())]

    def _run_in_thread(cmd: List[str], on_done) -> None:
        btn_scan.config(state=tk.DISABLED)
        btn_report.config(state=tk.DISABLED)
        status_var.set("Working…")

        def _worker():
            try:
                subprocess.run(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception as exc:
                root.after(0, lambda: status_var.set(f"Error: {exc}"))
                root.after(0, lambda: btn_scan.config(state=tk.NORMAL))
                root.after(0, lambda: btn_report.config(state=tk.NORMAL))
                return
            root.after(0, on_done)

        _th.Thread(target=_worker, daemon=True).start()

    def _after_scan() -> None:
        SCANNING_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DROP_DIR.mkdir(parents=True, exist_ok=True)
        moved = []
        for f in list(SCANNING_DIR.iterdir()):
            if f.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}:
                dst = REPORT_DROP_DIR / f.name
                try:
                    f.rename(dst)
                    moved.append(dst.name)
                except Exception:
                    pass
        if moved:
            status_var.set("Scan ready — click Grade Report to grade and save.")
        else:
            status_var.set("Scan done — no image found. Try again.")
        btn_scan.config(state=tk.NORMAL)
        btn_report.config(state=tk.NORMAL)

    def _after_report() -> None:
        nonlocal cards
        cards = _load_cards()
        _populate()
        n = len(cards)
        status_var.set(f"Graded! {n} card{'s' if n != 1 else ''} in portfolio.")
        btn_scan.config(state=tk.NORMAL)
        btn_report.config(state=tk.NORMAL)

    def _run_scan() -> None:
        _run_in_thread(_exe_cmd + ["--scan"], _after_scan)

    def _run_report() -> None:
        _run_in_thread(_exe_cmd + ["--report"], _after_report)

    btn_scan   = tk.Button(bar, text="Scan Card",    command=_run_scan,   **btn_style)
    btn_report = tk.Button(bar, text="Grade Report", command=_run_report, **btn_style)
    btn_scan.pack(side=tk.LEFT, padx=(0, 6))
    btn_report.pack(side=tk.LEFT)

    root.mainloop()


def _open_db() -> sqlite3.Connection:
    """Open portfolio DB, migrate schema if needed, return open connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    cols = {r[1] for r in con.execute("PRAGMA table_info(portfolio)")}
    outdated = cols and (
        any(c in cols for c in ("set_code", "sleeve_detected", "edge_grade", "qty"))
        or "scan_index" not in cols
    )
    if outdated:
        con.execute("DROP TABLE portfolio")
        logging.info("Portfolio DB: dropped outdated schema, recreating")
    con.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            card_name       TEXT,
            card_number     TEXT,
            hp              TEXT,
            scan_index      INTEGER DEFAULT 1,
            probable_grade  INTEGER,
            grade_band      TEXT,
            centering_lr    TEXT,
            centering_tb    TEXT,
            centering_grade INTEGER,
            corner_tl       REAL,
            corner_tr       REAL,
            corner_br       REAL,
            corner_bl       REAL,
            corner_grade    INTEGER,
            is_holo         INTEGER DEFAULT 0,
            print_line      INTEGER DEFAULT 0,
            limiting_factor TEXT,
            remediation     TEXT,
            ocr_confidence  TEXT,
            image_path      TEXT,
            scan_file       TEXT,
            date_scanned    TEXT
        )
    """)
    con.commit()
    return con


def _insert_portfolio_db(results: List["CardResult"]) -> int:
    """
    Always insert each graded card as a new row.
    Duplicate scans of the same card get scan_index = 1, 2, 3 …
    Returns total row count.
    """
    con = _open_db()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for r in results:
        idr = r.identification or IDResult()
        gr  = r.grade          or GradeAssessment()
        cen = r.centering      or CenteringResult()
        cor = r.corners        or CornerResult()
        hol = r.holo           or HoloResult()
        pl  = r.print_lines    or PrintLineResult()
        if not idr.card_name:
            continue
        # Compute next scan_index for this (name, number) pair
        existing = con.execute(
            "SELECT MAX(scan_index) FROM portfolio WHERE card_name=? AND card_number=?",
            (idr.card_name, idr.card_number),
        ).fetchone()[0]
        scan_idx = (existing or 0) + 1
        con.execute("""
            INSERT INTO portfolio (
                card_name, card_number, hp, scan_index,
                probable_grade, grade_band,
                centering_lr, centering_tb, centering_grade,
                corner_tl, corner_tr, corner_br, corner_bl, corner_grade,
                is_holo, print_line, limiting_factor, remediation,
                ocr_confidence, image_path, scan_file, date_scanned
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            idr.card_name, idr.card_number, idr.hp, scan_idx,
            gr.probable_grade, gr.grade_band,
            cen.lr_ratio, cen.tb_ratio, cen.centering_grade,
            cor.tl_whitening, cor.tr_whitening, cor.br_whitening, cor.bl_whitening,
            cor.corner_grade,
            int(hol.is_holo), int(pl.flagged),
            gr.limiting_factor, " | ".join(gr.remediation),
            idr.ocr_confidence, r.corrected_img_path, r.filename, now,
        ))
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM portfolio").fetchone()[0]
    con.close()
    return total


def write_outputs(results: List[CardResult], out_dir: Path, print_summary: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(r) for r in results]

    with open(out_dir / CSV_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)

    with open(out_dir / JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # ── Portfolio DB (always insert, never overwrite — duplicates get scan_index++) ──
    card_count = _insert_portfolio_db(results)

    # ── Terminal summary ───────────────────────────────────────────────────
    if not print_summary:
        return
    grade_counts = {}
    for r in results:
        g = r.grade.probable_grade if r.grade else 0
        grade_counts[g] = grade_counts.get(g, 0) + 1

    named   = sum(1 for r in results if r.identification and r.identification.card_name)
    errors  = sum(1 for r in results if r.error)
    holos   = sum(1 for r in results if r.holo and r.holo.is_holo)

    print(f"\n{'='*68}")
    print(f"  Cards processed   : {len(results)}")
    print(f"  Identified (OCR)  : {named}/{len(results)}")
    print(f"  Holo/foil cards   : {holos}/{len(results)}")
    print(f"  Errors            : {errors}")
    print(f"  Portfolio DB      : {DB_PATH}  ({card_count} cards total)")
    print(f"  Card images       : {IMAGES_DIR}")
    print(f"  CSV               : {out_dir / CSV_NAME}")

    print(f"\n  GRADE DISTRIBUTION  (centering + corners — surface unverified)")
    for g in sorted(grade_counts, reverse=True):
        label = f"PSA {g}" if g else "Error"
        bar   = "|" * min(grade_counts[g], 40)
        print(f"    {label:12s} {grade_counts[g]:>4}  {bar}")

    print(f"\n  ALL CARDS")
    print(f"  {'#':>4}  {'Card Name':<26}  {'#':>7}  {'HP':>4}  {'Holo':>5}  {'Grade Band':<36}  Limit")
    print(f"  {'-'*4}  {'-'*26}  {'-'*7}  {'-'*4}  {'-'*5}  {'-'*36}  {'-'*14}")
    for r in sorted(results, key=lambda x: -(x.grade.probable_grade if x.grade else 0)):
        gr   = r.grade or GradeAssessment()
        idr  = r.identification or IDResult()
        ho   = r.holo or HoloResult()
        name = idr.card_name or r.filename[:22]
        holo_tag = "HOLO " if ho.is_holo else "     "
        print(f"  {r.card_index:>4}  {name:<26}  {idr.card_number:>7}  {idr.hp:>4}  "
              f"{holo_tag}  {gr.grade_band:<36}  {gr.limiting_factor}")
    print(f"{'='*68}")


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER INTEGRATION  (Windows WIA via PowerShell subprocess — no extra deps)
# ══════════════════════════════════════════════════════════════════════════════

def _run_ps(script: str, timeout: int = 15) -> Tuple[int, str, str]:
    """Run a PowerShell script string. Returns (returncode, stdout, stderr)."""
    r = subprocess.run(
        ["powershell", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _detect_scanner() -> dict:
    """
    Query WIA for the first connected scanner.  Match against SCANNER_PROFILES
    by substring (case-insensitive).  Returns the matched profile or _DEFAULT_PROFILE.
    Also prints a one-line banner so the user knows what was found.
    """
    script = r"""
$mgr = New-Object -ComObject WIA.DeviceManager
$infos = $mgr.DeviceInfos
for ($i = 1; $i -le $infos.Count; $i++) {
    $d = $infos.Item($i)
    if ($d.Type -eq 1) {
        $d.Properties.Item("Name").Value
        break
    }
}
"""
    try:
        rc, out, err = _run_ps(script, timeout=10)
    except Exception:
        return _DEFAULT_PROFILE.copy()

    if rc != 0 or not out:
        return _DEFAULT_PROFILE.copy()

    detected_name = out.strip()
    for key, profile in SCANNER_PROFILES.items():
        if key.lower() in detected_name.lower():
            matched = profile.copy()
            matched["detected_name"] = detected_name
            return matched

    # Known scanner but no profile entry — return default with name attached
    p = _DEFAULT_PROFILE.copy()
    p["detected_name"] = detected_name
    return p


def _apply_scanner_profile(profile: dict) -> None:
    """
    Override global pixel-dimension constants based on the scanner profile.
    Percentage/fraction constants are left unchanged — only pixel counts scale.
    OCR_SCALE is divided by the same factor so the absolute OCR-crop size
    stays constant regardless of internal resolution (avoids easyocr confusion
    on over-large crops).
    """
    global TARGET_CARD_W, TARGET_CARD_H, CORNER_CROP_PX
    global PRINT_LINE_MIN_LENGTH, CORNER_SHARPNESS_MIN_DIST
    global OCR_SCALE, _SCANNER

    _SCANNER = profile
    scale = profile.get("scale", 1.0)
    TARGET_CARD_W             = int(630 * scale)    # base 630  px
    TARGET_CARD_H             = int(880 * scale)    # base 880  px
    CORNER_CROP_PX            = int(60  * scale)    # base 60   px
    PRINT_LINE_MIN_LENGTH     = int(80  * scale)    # base 80   px
    CORNER_SHARPNESS_MIN_DIST = int(5   * scale)    # base 5    px
    OCR_SCALE                 = max(1, round(4 / scale))  # keep abs crop size ~const


def _twain_scan(out_path: Path) -> bool:
    """
    Trigger a TWAIN scan via the 32-bit Python bridge (Canon ScanGear).
    Supports 1200/2400 DPI where the WIA driver caps at 600.
    Returns True on success, False on any failure.
    """
    if not _PY32.exists() or not _TWAIN_BRG.exists():
        logging.warning("TWAIN bridge not found — falling back to WIA")
        return False

    dpi         = _SCANNER.get("twain_dpi", 1200)
    zone_mm     = _SCANNER.get("zone_mm")
    zone_center = _SCANNER.get("zone_center", False)
    bed_mm      = _SCANNER.get("bed_mm")
    bmp_tmp     = out_path.with_suffix(".bmp")

    # Compute TWAIN frame (inches) for centred zone
    frame_args: List[str] = []
    if zone_mm and zone_center and bed_mm:
        zw, zh = zone_mm
        bw, bh = bed_mm
        left   = (bw - zw) / 2 / 25.4
        top    = (bh - zh) / 2 / 25.4
        right  = left + zw / 25.4
        bottom = top  + zh / 25.4
        frame_args = ["--frame", f"{left:.4f},{top:.4f},{right:.4f},{bottom:.4f}"]
        logging.info(f"TWAIN zone: {zw}×{zh} mm centred  frame=({left:.3f},{top:.3f},{right:.3f},{bottom:.3f})\"")

    logging.info(f"TWAIN scan: {dpi} DPI  → {bmp_tmp.name}")
    try:
        result = subprocess.run(
            [str(_PY32), str(_TWAIN_BRG), str(bmp_tmp), str(dpi)] + frame_args,
            capture_output=True, text=True, timeout=300,
        )
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        if stderr:
            logging.info(f"TWAIN bridge: {stderr}")
        if result.returncode != 0 or not stdout.startswith("OK:"):
            logging.error(f"TWAIN bridge failed: {stdout} {stderr}")
            return False

        bgr = cv2.imread(str(bmp_tmp))
        if bgr is None:
            logging.error("TWAIN scan: BMP unreadable")
            return False
        cv2.imwrite(str(out_path), bgr)
        return True

    except subprocess.TimeoutExpired:
        logging.error("TWAIN scan timed out")
        return False
    except Exception as e:
        logging.error(f"TWAIN scan exception: {e}")
        return False
    finally:
        try:
            bmp_tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _wia_scan(out_path: Path) -> bool:
    """
    Trigger a WIA scan using the active _SCANNER profile.
    Saves the result as a PNG to out_path.
    Returns True on success, False on any failure.

    Settings applied:
      - DPI from profile["scan_dpi"]
      - Colour mode from profile["color_wia"]
      - Scan zone from profile["zone_mm"] (top-left anchor, None = full bed)
      - Output: BMP (always-supported WIA format) → converted to PNG in Python
    """
    dpi         = _SCANNER.get("scan_dpi", 600)
    color_wia   = _SCANNER.get("color_wia", 3)
    zone_mm     = _SCANNER.get("zone_mm")
    zone_center = _SCANNER.get("zone_center", False)
    bed_mm      = _SCANNER.get("bed_mm")

    bmp_tmp = out_path.with_suffix(".bmp")

    # Compute extent pixels from zone_mm at the target DPI.
    # WIA Transfer() returns null if extent properties are left unset,
    # so we always set them explicitly — full bed uses the property's SubTypeMax.
    if zone_mm:
        zw, zh = zone_mm
        w_px = int(zw / 25.4 * dpi)
        h_px = int(zh / 25.4 * dpi)
        if zone_center and bed_mm:
            bw, bh = bed_mm
            x_start = int((bw - zw) / 2 / 25.4 * dpi)
            y_start = int((bh - zh) / 2 / 25.4 * dpi)
        else:
            x_start, y_start = 0, 0
        extent_block = (
            f"$item.Properties.Item('6149').Value = {x_start}\n"
            f"$item.Properties.Item('6150').Value = {y_start}\n"
            f"$item.Properties.Item('6151').Value = {w_px}\n"
            f"$item.Properties.Item('6152').Value = {h_px}\n"
        )
    else:
        # Full bed: read max extent from scanner property ranges (SubType 2 = Range)
        extent_block = (
            "$p6151 = $item.Properties.Item('6151')\n"
            "$p6152 = $item.Properties.Item('6152')\n"
            "$item.Properties.Item('6149').Value = 0\n"
            "$item.Properties.Item('6150').Value = 0\n"
            "if ($p6151.SubType -eq 2) { $item.Properties.Item('6151').Value = $p6151.SubTypeMax } "
            "else { $item.Properties.Item('6151').Value = 5100 }\n"
            "if ($p6152.SubType -eq 2) { $item.Properties.Item('6152').Value = $p6152.SubTypeMax } "
            "else { $item.Properties.Item('6152').Value = 7000 }\n"
        )

    script = f"""
$ErrorActionPreference = 'Stop'
$mgr = New-Object -ComObject WIA.DeviceManager
$infos = $mgr.DeviceInfos
$dev = $null
for ($i = 1; $i -le $infos.Count; $i++) {{
    if ($infos.Item($i).Type -eq 1) {{ $dev = $infos.Item($i).Connect(); break }}
}}
if ($null -eq $dev) {{ throw "No scanner found" }}
$item = $dev.Items.Item(1)
$item.Properties.Item("6147").Value = {dpi}
$item.Properties.Item("6148").Value = {dpi}
$item.Properties.Item("4103").Value = {color_wia}
{extent_block}
$fmt = "{{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}}"
$img = $item.Transfer($fmt)
$img.SaveFile("{bmp_tmp}")
Write-Output "OK"
"""
    try:
        logging.info(f"WIA scan: {dpi} DPI  zone={zone_mm}  → {bmp_tmp.name}")
        rc, out, err = _run_ps(script, timeout=120)
        if rc != 0 or "OK" not in out:
            logging.error(f"WIA scan failed: {err or out}")
            return False

        bgr = cv2.imread(str(bmp_tmp))
        if bgr is None:
            logging.error("WIA scan: BMP unreadable")
            return False
        cv2.imwrite(str(out_path), bgr)
        return True

    except subprocess.TimeoutExpired:
        logging.error("WIA scan timed out")
        return False
    except Exception as e:
        logging.error(f"WIA scan exception: {e}")
        return False
    finally:
        if bmp_tmp.exists():
            try: bmp_tmp.unlink()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN / CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global BINDER_ROWS, BINDER_COLS
    log_file = BASE_DIR / "card_grader.log"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _fh  = logging.FileHandler(str(log_file), encoding="utf-8")
    _fh.setFormatter(_fmt)
    _ch  = logging.StreamHandler(sys.stderr)
    _ch.setFormatter(_fmt)
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(_fh)
    logging.root.addHandler(_ch)

    # ── 1. Scanner detection + profile application ─────────────────────────
    scanner = _detect_scanner()
    _apply_scanner_profile(scanner)

    name    = scanner.get("detected_name") or scanner.get("model", "none")
    _twain_available = bool(scanner.get("twain_dpi") and _PY32.exists() and _TWAIN_BRG.exists())
    dpi_str = f"{scanner.get('scan_dpi', '?')} DPI" + (" (TWAIN/ScanGear)" if _twain_available else " (WIA)")
    scale   = scanner.get("scale", 1.0)
    zone    = scanner.get("zone_mm")
    zc      = scanner.get("zone_center", False)
    zone_str = (f"{zone[0]}×{zone[1]} mm (centred)" if (zone and zc) else
                f"{zone[0]}×{zone[1]} mm" if zone else "full bed")

    print(f"\n{'═'*66}")
    print(f"  Pokémon Card Grader")
    print(f"{'═'*66}")
    if scanner.get("detected_name"):
        print(f"  Scanner  : {name}  ·  {dpi_str}  ·  zone {zone_str}")
    else:
        print(f"  Scanner  : {name}  (no profile — using defaults)")
    print(f"{'─'*66}")

    # ── 2. OCR init ────────────────────────────────────────────────────────
    _init_ocr()

    # ── 3. CLI ─────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description=(
            "Pokemon card pre-grading — PSA grade estimate + remediation checklist.\n"
            f"Scanner  : {name}\n"
            f"Input    : {INPUT_DIR}\n"
            f"Output   : {OUTPUT_DIR}\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--scan",      action="store_true",
                     help=f"Scan card → {SCANNING_DIR.name}\\ (quick grade, no report)")
    src.add_argument("--report",    action="store_true",
                     help=f"Process {REPORT_DROP_DIR.name}\\, grade card, save to DB + images\\")
    src.add_argument("--portfolio", nargs="?", const="", metavar="SEARCH",
                     help="Show all graded cards in portfolio DB (optional name/number filter)")
    src.add_argument("--gui",       action="store_true",
                     help="Open card portfolio viewer GUI")
    src.add_argument("--image",     type=Path, help="Single card image file")
    src.add_argument("--imgdir",    type=Path, help=f"Folder of images (default: {INPUT_DIR})")
    src.add_argument("--pdf",       type=Path, help="Single binder-scan PDF (3×3 grid/page)")
    src.add_argument("--pdfdir",    type=Path, help="Folder of binder-scan PDFs")
    ap.add_argument("--outdir",       type=Path, default=OUTPUT_DIR)
    ap.add_argument("--binder-rows",  type=int,  default=BINDER_ROWS)
    ap.add_argument("--binder-cols",  type=int,  default=BINDER_COLS)
    args = ap.parse_args()
    BINDER_ROWS = args.binder_rows
    BINDER_COLS = args.binder_cols

    out     = args.outdir
    results: List[CardResult] = []

    # ── 4. --portfolio: show / search DB records ─────────────────────────────
    if args.portfolio is not None:
        _show_portfolio(args.portfolio)
        return

    # ── 4a. --gui: open card viewer ───────────────────────────────────────────
    if args.gui:
        _launch_gui()
        return

    # ── 4b. --scan: scan → scanning\ folder, quick grade only, NO report ─────
    if args.scan:
        if not scanner.get("detected_name"):
            print("\nERROR: No scanner detected. Connect scanner and retry.")
            return
        SCANNING_DIR.mkdir(parents=True, exist_ok=True)
        # Clear previous scan attempts so only the current card's scan remains
        for old in SCANNING_DIR.glob("*"):
            if old.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}:
                old.unlink()
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_out = SCANNING_DIR / f"scan_{stamp}.png"
        print(f"\n  Place card FACE DOWN (art side touching glass) anywhere on the glass.")
        print("  Scanning in 3 seconds...")
        time.sleep(3)
        print(f"  Scanning at {dpi_str}  zone {zone_str}  — please wait...")
        if _SCANNER.get("twain_dpi") and _PY32.exists() and _TWAIN_BRG.exists():
            ok = _twain_scan(scan_out)
            if not ok:
                logging.info("TWAIN failed — retrying with WIA at 600 DPI")
                ok = _wia_scan(scan_out)
        else:
            ok = _wia_scan(scan_out)
        if not ok:
            print("\nERROR: Scan failed — check scanner connection and try again.")
            return
        result = process_card_image(cv2.imread(str(scan_out)), scan_out.stem, out)
        result.filename   = scan_out.name
        result.card_index = 1
        results.append(result)
        # --scan is preview only — no DB insert; --report handles final save

        gr  = result.grade or GradeAssessment()
        idr = result.identification or IDResult()
        name_str = idr.card_name or "Unknown"
        num_str  = idr.card_number or "—"
        print(f"\n  {'─'*62}")
        print(f"  Card       : {name_str}  #{num_str}")
        print(f"  Grade est. : PSA {gr.probable_grade}  ({gr.grade_band or 'ungraded'})")
        if gr.limiting_factor:
            print(f"  Limited by : {gr.limiting_factor}")
        cent = result.centering
        if cent:
            print(f"  Centering  : L {cent.left_pct:.1f}%  R {cent.right_pct:.1f}%  "
                  f"T {cent.top_pct:.1f}%  B {cent.bottom_pct:.1f}%")
        print(f"  {'─'*62}")
        print(f"\n  Happy with it?")
        print(f"    1. Drag  {scan_out.name}  to  {REPORT_DROP_DIR}")
        print(f"    2. Run   card_grader --report")
        print(f"\n  Not happy? Run card_grader --scan to try again.")
        return

    # ── 4c. --report: process report_drop\, generate PDF, move to complete\ ──
    elif args.report:
        REPORT_DROP_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        drop_files = sorted(
            f for f in REPORT_DROP_DIR.iterdir()
            if f.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        if not drop_files:
            print(f"\n  report_drop\\ is empty — drag a scan there first.")
            print(f"  Folder: {REPORT_DROP_DIR}")
            return
        for i, scan_file in enumerate(drop_files, 1):
            print(f"\n  Processing [{i}/{len(drop_files)}]: {scan_file.name}")
            result = process_card_image(cv2.imread(str(scan_file)), scan_file.stem, out)
            result.filename   = scan_file.name
            result.card_index = i
            results.append(result)
        # Grade, save compressed images to images\, record everything in DB — no PDF
        write_outputs(results, out, print_summary=False)

        print(f"\n  {'─'*62}")
        for result in results:
            gr  = result.grade or GradeAssessment()
            idr = result.identification or IDResult()
            lbl = idr.card_name or result.filename
            img = Path(result.corrected_img_path).name if result.corrected_img_path else "—"
            print(f"  ✓  {lbl}  →  PSA {gr.probable_grade}  ({gr.grade_band or 'ungraded'})")
            print(f"     Image: {img}")
            # Delete raw scan — data is in DB, compressed image is in images\
            src_scan = REPORT_DROP_DIR / result.filename
            if src_scan.exists():
                src_scan.unlink()

        # Clear any leftover scans in scanning\
        for old in SCANNING_DIR.glob("*"):
            if old.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}:
                old.unlink()

        print(f"\n  Portfolio DB : {DB_PATH}")
        print(f"  Card images  : {IMAGES_DIR}")
        print(f"\n  Ready to scan the next card.")
        print(f"  {'─'*62}")

        return

    # ── 5. Normal file / folder sources ────────────────────────────────────
    elif args.image:
        result = process_card_image(cv2.imread(str(args.image)), args.image.stem, out)
        result.filename = args.image.name; result.card_index = 1
        results.append(result)

    elif args.imgdir:
        results = process_image_folder(args.imgdir, out)

    elif args.pdf:
        results = process_binder_pdf(args.pdf, out)

    elif args.pdfdir:
        pdfs = sorted(f for f in args.pdfdir.iterdir() if f.suffix.lower() == ".pdf")
        n = 1
        for pdf in pdfs:
            logging.info(f"PDF: {pdf.name}")
            batch = process_binder_pdf(pdf, out, start_n=n)
            results.extend(batch)
            n += len(batch)

    else:
        # No arguments — open GUI by default (double-click friendly)
        _launch_gui()
        return

    write_outputs(results, out)


if __name__ == "__main__":
    import traceback as _tb
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Cancelled.")
    except Exception as _exc:
        _crash = BASE_DIR / "card_grader_crash.txt"
        _msg   = _tb.format_exc()
        try:
            _crash.parent.mkdir(parents=True, exist_ok=True)
            _crash.write_text(_msg, encoding="utf-8")
        except Exception:
            pass
        print(f"\nCRASH — {_exc}")
        print(f"Full traceback saved to: {_crash}")
        print(_msg)
        raise SystemExit(1)
