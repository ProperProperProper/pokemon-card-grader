# Pokemon Card Pre-Grader

A fully offline, scanner-integrated Python tool for pre-grading Pokémon cards before PSA submission. Produces a probable PSA grade (7–10), per-axis metrics, holo/foil detection, and a detailed PDF report — all from a single flat scan.

---

## What It Does

| Feature | Detail |
|---|---|
| **Centering** | Measures left/right and top/bottom border ratios to PSA standards |
| **Corners** | Estimates whitening % + Shi-Tomasi sharpness at all 4 corners |
| **Holo / Foil detection** | 3-signal classifier (saturation gradient, patch std, sparkle %) — works on full-art, reverse holo, and illustration rare cards |
| **Surface** | Print-line artefact detection via Sobel + HoughLinesP — always marked *unverified* (flat scan cannot detect foil scratches) |
| **OCR identification** | Card name, number, and HP via EasyOCR |
| **PDF report** | Plain-English grade explanation with per-axis tables, action checklist, and card image |
| **Portfolio DB** | SQLite database auto-updated after every scan |
| **Scanner integration** | Native WIA (600 DPI) + Canon ScanGear TWAIN (1200 DPI) via a 32-bit Python bridge |

---

## Grading Method

PSA grades cards on four axes. This tool measures three of them from a flat scan:

```
probable_grade = min(centering_grade, corner_grade)
```

Surface grade is **always unverified** — a flat scanner cannot detect scratches, scuffs, or foil damage. Examine the surface under raking (angled) light before submitting.

### Centering thresholds

| Grade | Worse-side % |
|---|---|
| PSA 10 | ≤ 55% |
| PSA 9  | ≤ 60% |
| PSA 8  | ≤ 65% |
| PSA 7  | ≤ 70% |

### Corner thresholds

| Grade | Max whitening % |
|---|---|
| PSA 10 | ≤ 5% |
| PSA 9  | ≤ 12% |
| PSA 8  | ≤ 20% |
| PSA 7  | ≤ 30% |

---

## Requirements

```
pip install opencv-python numpy Pillow easyocr reportlab
```

**Optional (for 1200 DPI TWAIN scanning on Canon scanners):**

The script automatically uses a 32-bit Python bridge to access Canon ScanGear via TWAIN.  
Run `setup_twain_bridge.py` to install it (requires internet access, one-time only).

---

## Usage

```bash
# Scan a card directly from connected scanner (Canon LiDE 300 auto-detected)
python card_grader.py --scan

# Grade a single image
python card_grader.py --image "C:/scans/card.jpg"

# Grade a folder of card images
python card_grader.py --imgdir "C:/scans/cards/"

# Grade cards from a binder PDF (3×3 grid per page)
python card_grader.py --pdf "C:/scans/binder.pdf"
python card_grader.py --pdfdir "C:/scans/"
```

---

## Scanner Setup (Canon LiDE 300)

The tool auto-detects the scanner via WIA and applies an optimised profile:

| Setting | Value |
|---|---|
| DPI | 1200 (via Canon ScanGear TWAIN) |
| Colour | RGB 24-bit |
| Scan zone | Full bed |
| Internal resolution | 1260 × 1760 px (2× standard card) |

**How to scan:**
1. Place card **face-down** (art side touching the glass), anywhere on the bed
2. Run `python card_grader.py --scan`
3. Results appear in `C:\Users\<you>\Pictures\poke card grader\output\`

For best holo texture capture, remove the scanner lid and scan in a dimly lit room.

---

## Output

After each run, outputs are written to the configured output directory:

```
output/
  cards/               Corrected card images (perspective-corrected, auto-oriented)
  reports/             PDF grade reports (one per card)
  card_results.csv     All metrics in spreadsheet form
  card_results.json    Same data in JSON
card_grader_portfolio.db   SQLite portfolio database
```

### PDF Report Contents

Each card gets its own PDF with:
- Card identity (name, number, HP, holo status)
- Corrected card scan image
- **Grade verdict** in large coloured text with plain-English explanation
- Centering table (L/R and T/B splits with PSA benchmarks)
- Corner table (whitening % and sharpness per corner)
- Holo/foil detection result with confidence level
- Surface verification reminder
- Action checklist — everything needed before submitting to PSA

---

## Holo Detection

Three signals are measured from the artwork zone (top 8%–60% of card):

| Signal | How it works | Non-holo typical | Holo typical |
|---|---|---|---|
| `sat_grad` | Sobel gradient on HSV saturation channel | ~60–80 | ~90–180 |
| `local_sat_std` | Mean std across 16×16 saturation patches | ~20–28 | ~30–50 |
| `sparkle_pct` | % of artwork pixels with V > 235 | ~0.0% | ~0.4%+ |

A card is classified holo when **≥ 2 of 3 signals** exceed their thresholds.  
Confidence: `high` (3/3), `medium` (2/3), `low` (1/3).

Holo print lines (embossed foil texture) are excluded from surface analysis to avoid false positives.

---

## Auto-Orientation

The tool detects upside-down cards using a 3-signal majority vote:
1. White pixel fraction in top vs bottom half (classic cards)
2. Light pixel fraction in top/bottom 8% strips (card number strip)
3. Saturation comparison in top/bottom 8% strips (type band vs card number)

Requires ≥ 2/3 votes to trigger a 180° rotation. Works on full-art, gold, and illustration rare cards that lack a traditional white text box.

---

## Architecture

Single-file design (`card_grader.py`) — all logic in one place, no package structure needed.

| Component | Approach |
|---|---|
| Card detection | 3-strategy pipeline: Canny quad → brightness threshold → colour band |
| Rotation correction | Rotation-only (`_rotate_and_crop`) — perspective warp intentionally disabled (distorts border measurements) |
| OCR | EasyOCR (self-contained, ~100 MB model download on first run) |
| Scanner control | WIA via PowerShell subprocess (no native dependencies) + TWAIN via 32-bit bridge |
| Database | SQLite with auto-migration |

---

## References

- [crimsonthinker/psa_pokemon_cards](https://github.com/crimsonthinker/psa_pokemon_cards) — 4-axis grading framework  
- [rthorst/mint_condition](https://github.com/rthorst/mint_condition) — feature importance for card grading  
- [NickPiscitelli/pokemon-card-analyzer](https://github.com/NickPiscitelli/pokemon-card-analyzer) — border measurement approach  
- ScienceDirect automated corner grading paper — Shi-Tomasi + Hough corner detection  

---

## Disclaimer

Grade estimates from this tool are for guidance only. PSA final grades may differ based on factors not detectable by flat scanning (surface texture, foil condition, print defects). Always verify under raking light before submitting.
