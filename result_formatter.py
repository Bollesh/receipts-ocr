from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from typing import Optional
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────
 
# Dates: DD/MM/YYYY, MM-DD-YY, YYYY-MM-DD, DD MMM YYYY, etc.
DATE_PATTERNS = [
    r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',
    r'\b\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}\b',
    r'\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s,]+\d{2,4}\b',
    r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s,]+\d{1,2}[\s,]+\d{2,4}\b',
]
 
# Prices: optional currency symbol, digits, optional spaces around separator, two decimals.
# Handles OCR artifacts like "86 . 35" or "82 . 75" in addition to normal "86.35".
PRICE_PATTERN   = re.compile(r'(?:RS\.?|INR|USD|\$|£|€|RM)?\s*(\d{1,6}\s*[\.,]\s*\d{2})\b')
 
# Total-line keywords (ordered by priority)
TOTAL_KEYWORDS  = ["GRAND TOTAL", "TOTAL AMOUNT", "NET AMOUNT", "NET TOTAL",
                   "AMOUNT DUE", "TOTAL DUE", "SUBTOTAL", "TOTAL", "AMOUNT", "NET"]
 
# Lines that should never be mistaken for the store name
STORE_BLACKLIST = re.compile(
    r'\b(INVOICE|RECEIPT|TAX|GST|VAT|TEL|FAX|DATE|TIME|NO\.?|CASH|BILL|'
    r'PHONE|EMAIL|WWW|HTTP|CUSTOMER|TABLE|SERVER|ORDER|VOID|COPY|DUPLICATE)\b'
)
 
# Lines that look like metadata, not item names
ITEM_BLACKLIST  = re.compile(
    r'\b(TOTAL|SUBTOTAL|TAX|GST|VAT|DISCOUNT|CHANGE|CASH|CARD|TIP|'
    r'THANK|WELCOME|VISIT|CALL|MEMBER|LOYALTY|POINTS|'
    r'ROUNDING|PROMO|SAVINGS|SAVING|PROMOTION|REBATE|'
    r'MASTERCARD|VISA|AMEX|DEBIT|CREDIT|PAYMENT|TEND|'
    r'DATE|TIME|TEL|FAX|WWW|HTTP|EMAIL|PHONE)\b'
)
 
# Item name must have at least 2 meaningful characters after stripping price/symbols
MIN_ITEM_NAME_LEN = 2
 
# Minimum OCR confidence to even consider a token
MIN_OCR_PROB = 0.30
 
# Row-grouping tolerance: lines within this many pixels share the same row
ROW_TOLERANCE_PX = 12
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Internal data model
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class Token:
    """One EasyOCR detection."""
    bbox:  list
    text:  str
    prob:  float
    y_mid: float = field(init=False)
 
    def __post_init__(self):
        ys = [pt[1] for pt in self.bbox]
        self.y_mid = (min(ys) + max(ys)) / 2
 
    @property
    def upper(self) -> str:
        return self.text.strip().upper()
 
 
@dataclass
class Row:
    """Tokens that share approximately the same vertical position."""
    tokens: list[Token]
 
    @property
    def text(self) -> str:
        return " ".join(t.text.strip() for t in self.tokens)
 
    @property
    def upper(self) -> str:
        return self.text.upper()
 
    @property
    def avg_prob(self) -> float:
        return sum(t.prob for t in self.tokens) / len(self.tokens) if self.tokens else 0.0
 
    @property
    def max_prob(self) -> float:
        return max(t.prob for t in self.tokens) if self.tokens else 0.0
 
    @property
    def y_mid(self) -> float:
        return sum(t.y_mid for t in self.tokens) / len(self.tokens)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Grouping helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _group_into_rows(tokens: list[Token], tolerance: int = ROW_TOLERANCE_PX) -> list[Row]:
    """Cluster tokens that are on the same horizontal line."""
    if not tokens:
        return []
    sorted_tokens = sorted(tokens, key=lambda t: t.y_mid)
    rows: list[Row] = []
    current: list[Token] = [sorted_tokens[0]]
 
    for tok in sorted_tokens[1:]:
        if abs(tok.y_mid - current[-1].y_mid) <= tolerance:
            current.append(tok)
        else:
            # Sort row left-to-right before saving
            rows.append(Row(sorted(current, key=lambda t: t.bbox[0][0])))
            current = [tok]
    rows.append(Row(sorted(current, key=lambda t: t.bbox[0][0])))
    return rows
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Field extractors
# ─────────────────────────────────────────────────────────────────────────────
 
def _extract_price(text: str) -> Optional[str]:
    """Return the first well-formed price found in *text*, or None."""
    m = PRICE_PATTERN.search(text.upper())
    if not m:
        return None
    # Collapse internal spaces (e.g. "86 . 35" → "86.35") then normalise comma→dot
    raw = re.sub(r'\s', '', m.group(1)).replace(",", ".")
    # Sanity check: at least one integer digit before decimal
    parts = raw.split(".")
    if not parts[0].isdigit():
        return None
    # Reject implausibly large single-item prices (barcode digits fused with price)
    if float(raw) > 9999.99:
        return None
    return raw
 
 
def _extract_date(text: str) -> Optional[str]:
    """Return the first date-like string found in *text*, or None."""
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, text.upper())
        if m:
            return m.group().strip()
    return None
 
 
def _is_plausible_store_name(row: Row, row_index: int) -> bool:
    """
    Heuristics for deciding whether a row looks like a store/vendor name.
 
    Rules
    -----
    - Must appear in the top 20 % of the receipt (first few rows)
    - Must not match the blacklist keywords
    - Must not be a pure number or price
    - Must have at least 3 characters after stripping punctuation
    - OCR confidence >= 0.5
    """
    if row_index > 5:
        return False
    if STORE_BLACKLIST.search(row.upper):
        return False
    if _extract_price(row.text):
        return False
    if _extract_date(row.text):
        return False
    cleaned = re.sub(r'[^A-Z0-9\s]', '', row.upper).strip()
    if len(cleaned) < 3:
        return False
    if re.fullmatch(r'\d+', cleaned):   # Pure number
        return False
    if row.avg_prob < 0.50:
        return False
    return True
 
 
def _is_plausible_item(row: Row, price: str) -> bool:
    """Check that a row can reasonably represent a purchased item."""
    if ITEM_BLACKLIST.search(row.upper):
        return False
    name = row.upper.replace(price, "").strip()
    name = re.sub(r'[^A-Z0-9\s]', '', name).strip()
    if len(name) < MIN_ITEM_NAME_LEN:
        return False
    # Reject rows that are almost entirely numeric
    letters = re.sub(r'[^A-Z]', '', name)
    if len(letters) < 2:
        return False
    return True
 
 
def _confidence_for_field(
    ocr_prob: float,
    pattern_matched: bool,
    keyword_matched: bool = False,
    position_bonus: float = 0.0,
) -> float:
    """
    Weighted confidence score for a single extracted field.
 
    Components
    ----------
    ocr_prob        : raw EasyOCR probability  (weight 0.50)
    pattern_matched : regex validated the value (weight 0.30)
    keyword_matched : a strong heuristic keyword was nearby (weight 0.15)
    position_bonus  : extra credit for correct position on receipt (weight 0.05)
    """
    score = (
        0.50 * ocr_prob
        + 0.30 * float(pattern_matched)
        + 0.15 * float(keyword_matched)
        + 0.05 * position_bonus
    )
    return round(min(score, 1.0), 3)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main formatter
# ─────────────────────────────────────────────────────────────────────────────
 
def format_assignment_output(ocr_results: list[tuple]) -> dict:
    """
    Convert raw EasyOCR output into a structured, confidence-aware JSON dict.
 
    Parameters
    ----------
    ocr_results : list of (bbox, text, prob) tuples from EasyOCR
 
    Returns
    -------
    dict with keys: store_name, date, items, total_amount, flags
    """
 
    # ── 0. Filter out very low-confidence detections ─────────────────────────
    tokens = [
        Token(bbox, text, prob)
        for bbox, text, prob in ocr_results
        if prob >= MIN_OCR_PROB and text.strip()
    ]
 
    # ── 1. Group tokens into rows ─────────────────────────────────────────────
    rows = _group_into_rows(tokens)
 
    # ── 2. Initialise output ──────────────────────────────────────────────────
    output: dict = {
        "store_name":   {"value": None, "confidence": 0.0},
        "date":         {"value": None, "confidence": 0.0},
        "items":        [],
        "total_amount": {"value": None, "confidence": 0.0},
        "flags":        [],         # low-confidence field warnings
    }
 
    # Track all candidate totals so we can pick the highest-confidence one
    total_candidates: list[dict] = []
 
    # ── 3. Walk rows ──────────────────────────────────────────────────────────
    for row_idx, row in enumerate(rows):
        upper = row.upper
 
        # ── A. Store name (first plausible row near the top) ──────────────
        if output["store_name"]["value"] is None:
            if _is_plausible_store_name(row, row_idx):
                conf = _confidence_for_field(
                    ocr_prob=row.avg_prob,
                    pattern_matched=True,    # Heuristic passed
                    position_bonus=1.0 if row_idx == 0 else 0.5,
                )
                output["store_name"] = {"value": row.text.strip(), "confidence": conf}
 
        # ── B. Date ───────────────────────────────────────────────────────
        if output["date"]["value"] is None:
            date_val = _extract_date(upper)
            if date_val:
                keyword_near = bool(re.search(r'\bDATE\b', upper))
                conf = _confidence_for_field(
                    ocr_prob=row.avg_prob,
                    pattern_matched=True,
                    keyword_matched=keyword_near,
                )
                output["date"] = {"value": date_val, "confidence": conf}
 
        # ── C. Total amount (collect all candidates) ──────────────────────
        matched_keyword = next(
            (kw for kw in TOTAL_KEYWORDS if kw in upper), None
        )
        if matched_keyword:
            price_val = _extract_price(upper)
 
            # If price not on same line, check next row — but ONLY when that
            # row doesn't look like a payment/tax line.  Without this guard the
            # lookahead would grab the CASH or TAX amount whenever the real
            # total had a spaced decimal ("86 . 35") that failed to parse.
            _NEXT_ROW_SKIP = re.compile(
                r'\b(CASH|CHANGE|TAX|DEBIT|CREDIT|TEND|CARD|VISA|MASTERCARD|'
                r'AMEX|PAYMENT|TIP|ROUNDING)\b'
            )
            if price_val is None and row_idx + 1 < len(rows):
                next_row = rows[row_idx + 1]
                if not _NEXT_ROW_SKIP.search(next_row.upper):
                    price_val = _extract_price(next_row.upper)
                    next_prob = next_row.avg_prob if price_val else 0.0
                    blended_prob = (row.avg_prob + next_prob) / 2
                else:
                    blended_prob = row.avg_prob
            else:
                blended_prob = row.avg_prob
 
            if price_val:
                # Prioritise more specific keywords
                kw_priority = 1.0 - (TOTAL_KEYWORDS.index(matched_keyword) / len(TOTAL_KEYWORDS))
                conf = _confidence_for_field(
                    ocr_prob=blended_prob,
                    pattern_matched=True,
                    keyword_matched=True,
                    position_bonus=kw_priority * 0.5,
                )
                total_candidates.append({"value": price_val, "confidence": conf})
 
        # ── D. Item lines ─────────────────────────────────────────────────
        price_val = _extract_price(upper)
        if price_val and _is_plausible_item(row, price_val):
            # Clean item name: remove price, currency symbols, leading codes
            name = upper.replace(price_val, "")
            name = re.sub(r'^[\d\s#\.\-]+', '', name)          # strip leading item codes
            name = re.sub(r'\b\d{8,}\b', '', name)             # strip barcode-length numbers
            name = re.sub(r'\b(FS|SR|F|N|T)\b', '', name)      # strip tax/food-stamp flags
            name = re.sub(r'[^\w\s\-\/\(\)]', ' ', name)     # strip odd symbols
            name = re.sub(r'\s{2,}', ' ', name).strip()
 
            item_conf = _confidence_for_field(
                ocr_prob=row.avg_prob,
                pattern_matched=True,
            )
            output["items"].append({
                "name":       name,
                "price":      price_val,
                "confidence": item_conf,
            })
 
    # ── 4. Resolve best total ─────────────────────────────────────────────────
    if total_candidates:
        best = max(total_candidates, key=lambda c: c["confidence"])
        output["total_amount"] = best
    
    # ── 5. De-duplicate items (same name+price from overlapping rows) ─────────
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for item in output["items"]:
        key = (item["name"], item["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    output["items"] = deduped
 
    # ── 6. Remove items that duplicate the total ──────────────────────────────
    total_val = output["total_amount"].get("value")
    if total_val:
        output["items"] = [it for it in output["items"] if it["price"] != total_val]
 
    # ── 7. Flag low-confidence fields ─────────────────────────────────────────
    LOW_CONF_THRESHOLD = 0.60
    for field_name in ("store_name", "date", "total_amount"):
        field = output[field_name]
        if field["value"] is None:
            output["flags"].append(f"MISSING: {field_name}")
        elif field["confidence"] < LOW_CONF_THRESHOLD:
            output["flags"].append(
                f"LOW_CONFIDENCE: {field_name} ({field['confidence']:.2f})"
            )
 
    return output
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer (for debugging)
# ─────────────────────────────────────────────────────────────────────────────
 
def pretty_print(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))