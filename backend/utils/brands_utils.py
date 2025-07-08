import re
import json
from pathlib import Path
from functools import lru_cache
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

BRAND_FILE = Path(settings.BASE_DIR) / "utils" / "brands.json"

@lru_cache(maxsize=1)
def load_brand_data():
    """Load brands and exceptions from the JSON file once."""
    try:
        with BRAND_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
            
    except FileNotFoundError:
        logger.error("Brand JSON file not found at %s", BRAND_FILE)
        data = {"brands": [], "exceptions": []}
        
    return {
        "brands": {b.upper() for b in data.get("brands", [])},
        "exceptions": {e.upper() for e in data.get("exceptions", [])},
    }
    
def is_brand(term: str) -> bool:
    """
    Return True if `term` exactly matches a known brand (case-insensitive whole-word match).
    """
    
    term_up = term.strip().upper()
    data = load_brand_data()
    
    # if term_up in data["exceptions"]:
    #     return False
    # whole-word check via set membership
    return term_up in data["brands"]

def find_brands_in_text(text: str) -> list[str]:
    """
    Scan `text` for any occurrences of known brands, respecting word boundaries and excluding exceptions.
    Returns list of matched brand names.
    """
    data = load_brand_data()
    brands = data["brands"] - data["excptions"]
    # build a regex like r"\b(ANTARCT|ANTARC|LOPES|....)\b"
    pattern = r"\b(" + "|".join(re.escape(b) for b in sorted(brands, key=len, reverse=True)) + ")\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    
    # dedupe and uppercase
    return sorted({m.upper() for m in matches})

def preprocess_text(text: str) -> str:
    """
    Any additional text clean-up (e.g strip punctuation, normalzie whitespace).
    """
    # Example: remove leading dots, commas, etc.
    return re.sub(r"^[\.\,\s]+", "", text).strip()
