import re

KNOWN_METRICS = [
    "UN", "CX", "PT", "PCT", "PR", "CXT", "CT", "CXA" , "KG", "G", "M"
]

QUANTITY_RE = re.compile(r"""
                         \b(\d+\s?X\s?\d+(?:\s?[A-Za-z]+)?)\b   # matches '1X24', '1 X 24 500G'
                        |\b(\d+\s?(?:KG|G|ML|L|MT))\b          # matches '500G'
                         """, re.IGNORECASE | re.VERBOSE)

def extract_metric_and_quantity(text: str) -> tuple[str, str, str]:
    """
    Returns (cleaned_text, quantity).
    """
    q_match = QUANTITY_RE.search(text)
    quantity = q_match.group(0) if q_match else ""
    text = text.replace(quantity, "").strip()
    
    return text, quantity
