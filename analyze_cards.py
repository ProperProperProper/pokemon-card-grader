"""
Pokemon card analyser — sends each cropped card image to Claude claude-haiku-4-5
and returns structured identification + value estimate.

Usage:
    python analyze_cards.py

Output:
    C:\Users\User\Documents\card_analysis.json   — full results
    C:\Users\User\Documents\card_analysis.csv    — spreadsheet-friendly
    Prints a sorted list of cards worth >$1 raw at the end.

Set your API key first:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""

import anthropic
import base64
import json
import csv
import os
import re
import time
from pathlib import Path
from PIL import Image, ImageEnhance
import io

INDIVIDUAL_DIR = Path(r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User\7c19d49a-94d2-4fde-a0ae-d6fb66c927b5\scratchpad\individual")
OUT_JSON = Path(r"C:\Users\User\Documents\card_analysis.json")
OUT_CSV  = Path(r"C:\Users\User\Documents\card_analysis.csv")

PROMPT = """You are a Pokemon TCG expert. Look at this card image carefully.

Return ONLY a JSON object with these fields — no extra text:
{
  "blank": true/false,           // true if the slot is empty, a basic Energy card (no special art), or too blurry to read
  "name": "...",                 // full card name e.g. "Mega Lucario ex", "Boss's Orders"
  "set_code": "...",             // 2-4 letter set code if visible (MEG, SFA, TWM, POR, CRI, PBL, MEP, etc.) or null
  "card_number": "...",          // e.g. "077/132" or "081" for promos, or null if unreadable
  "rarity": "...",               // Common / Uncommon / Rare Holo / Double Rare / Special Illustration Rare / Black Star Promo / Special Energy / etc.
  "card_type": "...",            // Pokemon / Trainer / Energy
  "hp": null or integer,        // HP number for Pokemon, null for Trainers/Energy
  "pokemon_type": "...",         // Fire / Water / Grass / Lightning / Fighting / Psychic / Darkness / Metal / Colorless / Dragon / null
  "is_ex": true/false,           // true if the card name includes "ex" or "EX"
  "is_full_art": true/false,     // true if artwork bleeds edge-to-edge (Illustration Rare / Full Art style)
  "price_usd_estimate": 0.00,   // your best estimate of raw NM ungraded market price in USD
  "notes": "..."                 // anything notable: promo stamp, unusual art, special foil, etc. or ""
}

Key identifiers to look for:
- Card number (bottom left of card) — important for Secret/Illustration Rares (number > set total)
- Rarity symbol (bottom right): ◆=Common, ◆◆=Uncommon, ★=Rare Holo, ★★=Double Rare, ☆=Special
- "ex" in name = ex Pokemon (usually Double Rare or higher)
- Full-art cards have artwork that covers the entire card face
- Black Star Promos have a star+SWSH or similar stamp

If you cannot read the card clearly, still try — make your best guess and note uncertainty.
Only set blank=true for genuinely empty slots or plain basic Energy cards."""

def enhance_image(img_path: Path) -> bytes:
    """Upscale 3× and boost contrast/sharpness for better readability."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    img = img.resize((w * 3, h * 3), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(2.5)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def analyse_card(client: anthropic.Anthropic, img_path: Path) -> dict:
    image_bytes = enhance_image(img_path)
    b64 = base64.standard_b64encode(image_bytes).decode()

    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text",  "text": PROMPT}
            ]
        }]
    )

    raw = msg.content[0].text.strip()
    # Extract JSON even if Claude wraps it in markdown
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"blank": True, "name": "parse_error", "notes": raw[:200]}

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        print("  In PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        return

    client = anthropic.Anthropic(api_key=api_key)

    images = sorted(INDIVIDUAL_DIR.glob("card_*.png"))
    print(f"Found {len(images)} card images to analyse.")

    results = []
    # Resume support — load existing results if any
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            results = json.load(f)
        done_names = {r["file"] for r in results}
        images = [p for p in images if p.name not in done_names]
        print(f"Resuming — {len(images)} cards remaining.")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name} ... ", end="", flush=True)
        try:
            data = analyse_card(client, img_path)
        except Exception as e:
            data = {"blank": False, "name": "api_error", "notes": str(e)}

        data["file"] = img_path.name
        results.append(data)

        status = "BLANK" if data.get("blank") else data.get("name", "?")
        price  = data.get("price_usd_estimate", 0) or 0
        print(f"{status}  ${price:.2f}")

        # Save after every card so we can resume
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

        # Polite rate-limit (haiku is fast but let's not hammer it)
        time.sleep(0.3)

    # ── Write CSV ──────────────────────────────────────────────────────────
    fields = ["file","blank","name","set_code","card_number","rarity","card_type",
              "hp","pokemon_type","is_ex","is_full_art","price_usd_estimate","notes"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {OUT_JSON.name} and {OUT_CSV.name}")

    # ── Print valuable cards ───────────────────────────────────────────────
    valuable = [r for r in results if not r.get("blank") and (r.get("price_usd_estimate") or 0) >= 1.0]
    valuable.sort(key=lambda r: r.get("price_usd_estimate", 0), reverse=True)

    print(f"\n{'='*65}")
    print(f"  CARDS WORTH $1+ (raw NM) — {len(valuable)} found")
    print(f"{'='*65}")
    for r in valuable:
        num   = r.get("card_number") or "?"
        rarity= r.get("rarity") or ""
        price = r.get("price_usd_estimate", 0)
        print(f"  ${price:>6.2f}  {r['name']:<35} {num:<10} {rarity}")

    total_low  = sum((r.get("price_usd_estimate") or 0) for r in results if not r.get("blank"))
    non_blank  = sum(1 for r in results if not r.get("blank"))
    print(f"\n  Total non-blank cards: {non_blank}")
    print(f"  Estimated collection value: ~${total_low:.2f} USD (raw NM)")

if __name__ == "__main__":
    main()
