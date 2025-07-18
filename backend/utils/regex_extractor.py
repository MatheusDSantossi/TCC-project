import re
import unicodedata

from typing import List

pattern = re.compile(r"\[(.*?)\]")

def extract_product_list(text: str) -> List[str]:
    """
    Given an AI-preprocessed prompt that (we hope) has "...,p1,p2,p3,..." return [p1, p2, p3].
    """
    
    text = text.lower().strip()
    match = pattern.search(text)
    
    if not match:
        return []
    
    product_string = match.group(1)
    
    # Split on commas, strip out accents
    def clean(p):
        p = unicodedata.normalize("NFKD", p).encode("ascii", "ignore").decode("ascii")
        
        return p.strip()
    
    return [clean(p) for p in product_string.split(",") if clean(p)]
