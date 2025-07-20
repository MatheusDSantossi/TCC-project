import unicodedata
import re
from decimal import Decimal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import logging
from typing import List, Optional, Dict

from django.db import transaction
from django.db.models import Q

from chatbot.models import Product
from utils.brands_utils import find_brands_in_text
from utils.metrics_utils import extract_metric_and_quantity

import google.generativeai as genai

logger = logging.getLogger(__name__)

def find_business_sections(business_id: Optional[int] = None) -> List[str]:
    """
    Return a sorted list of distinct product sections.
    If business_id is provided, only sections for that business. However if it isn't provide, my product change based on the chatbot website url (I have to verify this)
    """
    qs = Product.objects.values_list("section", flat=True)
    
    if business_id is not None:
        qs = qs.filter(business_id=business_id)
        
        
    # Fetch all raw secitons strings
    raw_sections = list(qs)
    
    # Normalize: strip whitespace and ignore empty strings
    normalized = {
        sec.strip()
        for sec in raw_sections
        if sec and sec.strip()
    }
    
    # Sort case-insensitively
    sections = sorted(normalized, key=str.lower)
    
    logger.debug("Found sections (%d): %r", len(sections), sections)
    
    return sections
    
def find_section(text: str, business_id: Optional[int] = None):
    # Define the regular expression to match "section" followed by a letter or a number
    pattern = r'\bseção\s([a-zA-Z0-9]+)\b'
    
    # Search for the pattern in the input text (case-insensitive)
    match = re.search(pattern, text, re.IGNORECASE)
    
    print("section Text (find section): ", text)
    
    # if a match is found, return the matched section
    if match:
        
        section = match.group(1)
        
        section = "Seção " + section
        
        print("Section after normalize: ", section)
        
        qs = Product.objects.filter(section=section)
    
        if business_id is not None:
            qs = qs.filter(business_id=business_id)
            
        results = qs.values("name", "section")[:30]
    
        print("List of products (find section): ", list(results))
        
        return list(results)

    return None

def find_price(text, business_id: Optional[int] = None):
    # Define the regular expression to match a price in the form of $99.99
    pattern = r'\b\d{1,3}(?:[.,]\d{2})\b'
    
    # Search for the pattern in the input text
    match = re.search(pattern, text)
    
    # +- 5 cents of difference in the price
    delta = Decimal("0.05")
    
    # if a match is found, return the matched price
    if match:
        price = match.group()
        price_with_point = Decimal(price.replace(',', '.'))
                
        qs = Product.objects.filter(price__gte=price_with_point - delta, price__lte=price_with_point + delta)
        
        if business_id is not None:
            qs = qs.filter(business_id=business_id)
              
        results = qs.values_list("name", "price")[:30]
        
        print("Products find with the same price: ", list(results))
        
        return list(results)
        
    return None

def remove_accents(input_str):
    """Remove accents/diacritics from a string while preserving the base chracter"""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def find_brand(text: str, business_id: Optional[int]= None) -> Optional[List[Dict]]:
    """
    Look for known brand names in `text`. If found, return up to 30 matching products (name and brand). Returns None if no brand match.
    """
    
    print("Text (find_brand): ", text)
    
    normalize_text = remove_accents(text).upper()
    
    # Normalize & detect brands
    brands = find_brands_in_text(normalize_text)
    
    if not brands:
        return None


    brand = brands[0] # use the first match
    logger.debug("Brand match: %s in text: %r", brands, text)
    
    # Query via ORM
    qs = Product.objects.filter(brand__iexact=brand)
    
    if business_id is not None:
        qs = qs.filter(business_id=business_id)
        
    qs = qs.values("name", "brand")[:30]
    
    print("List of products (find brand): ", list(qs))
    
    return list(qs) or None
    
    
def find_quantity(text: str, business_id: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Extract quantity/unit information from text and find matching products. 
    Returns a list of product dictionaries or None if no matches found.
    """
    # Preprocess text
    normalized_text = remove_accents(text).upper()
    logger.debug(f"Processing quantity for text: {normalized_text}")
    print(f"Processing quantity for text: {normalized_text}")
    
    # Extract quantity and metric using regex
    metric, quantity = None, None

    # Regex info:
    #   - an "x" with another number and then some letters (e.g., "1 X 24 500G")
    #   - or a standalone number with a unit (e.g., "500G", "500ML", "500MT")
    quantity_pattern = r"\b\d+\s*(?:[Xx]\s*\d+\s*[A-Za-z]+|\s?(?:KG|G|ML|L|MT|CX|PT|PCT|UN|LTS|GR))\b"
    
    match = re.search(quantity_pattern, normalized_text)
    
    if not match:
        logger.debug("No quantity/unity pattern found")
        print("No quantity/unity pattern found")
        return None
    

    quantity_str = match.group().strip()
    
    # Split of trailing letters as metric
    metric_match = re.search(r"[A-Za-z]+$", quantity_str)
    
    metric = metric_match.group().upper() if metric_match else None
    quantity = quantity_str
    
    logger.debug("Extracted quantity=%r, metric=%r", quantity, metric)
    print(f"Extracted quantity={quantity}, and metric={metric}")
    
    # Build ORM query
    query = Q()
    
    if quantity:
        query |= Q(quantity__iexact=quantity)
    
    if metric:
        query |= Q(unit__iexact=metric)
    
    if not query:
        logger.debug("No valid quantity or metric found")
        
        return None    
    
    # Execute query
    qs = Product.objects.filter(query)
    
    if business_id:
        qs = qs.filter(business_id=business_id)
    
    print(f"Found before name filter {len(qs)} matching products")
    
    # TODO: Think about a efficient way to narrow by product name. Considering that the text contains more than the product name.
    name_filter = normalized_text.replace(quantity_str, "").strip()
    
    # if name_filter:
    #     qs = qs.filter(name__icontains=name_filter)
    
    results = qs.values("name", "quantity", "unit")[:30]
    
    if not results:
        logger.debug("No matching products found")
        print("No matching products found")
        return None
    
    logger.debug(f"Found {len(results)} matching products")
    print(f"Found {len(results)} matching products")
    return list(results)

# TODO: Fix this function because the return is not good
# def find_similar_products_using_input_embeddings(input_embed: List[float], threshold: float = 0.4, top_k: int = 20, business_id: Optional[int]= None) -> List[Dict]:
def find_similar_products_using_input_embeddings(user_text: str, embed_fn, threshold: float = 0.4, top_k: int = 20, business_id: Optional[int]= None) -> List[Dict]:
    """
    Given an input embedding, find the top_k most-similar products (similarity > threshold) and return their details + score.
    """
    
    # Extract the likely product phrase
    # prod_phrase = extract_product_chunk(user_text)
    
    # Hybrid pre-filter
    pre_qs = Product.objects
    
    if business_id is not None:
        qs = qs.filter(business_id=business_id)
    
    pre_qs = pre_qs.filter(name__icontains=user_text)[:200]
    
    rows = list(pre_qs.values_list("product_id", "embeddings"))
    
    if not rows:
        return []
    
    # Pull all IDs + embeddings in one go:
    # qs = Product.objects.values_list("product_id", "embeddings")
    
        
    # rows = list(qs) # list of (id, [float, ...])
    
    product_ids, embeddings = zip(*rows)
    embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
    
    # Compute similarity scores
    input_vec = np.array(embed_fn(find_section, genai, business_id, user_text), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(input_vec, embeddings)[0] 
    
    # Pick the ones above your cutoff and sort
    # idxs = np.argsort(sims)[::-1]
    # similar = [
    #     (product_ids[i], float(sims[i]))
    #     for i in idxs
    #     if sims[i] > threshold
    # ][:top_k]
    
    # Pick & fetch details
    similar = sorted(
        [(ids[i], float(sims[i])) for i in range(len(ids)) if sims[i] > threshold], key=lambda x: -x[1]
    )[:top_k]
    
    if not similar:
        return []
    
    # Fetch all those product details in one ORM hit
    # ids = [pid for pid, _ in similar]
    # prods = Product.objects.filter(product_id__in=ids).in_bulk(field_name="product_id")
    
    products = Product.objects.in_bulk([pid for pid, _ in similar], field_name="product_id")
    
    return [
        {
             "product_id": pid,
            "name": prod.name,
            "section": prod.section,
            "price": prod.price,
            "quantity": prod.quantity,
            "unit": prod.unit,
            "similarity": round(score,4),
        }
        for pid, score in similar if (prod:=products.get(pid))
    ]
    
    # Build your return list, preserving the similarity order
    # result = []
    # for pid, score in similar:
    #     p = prods.get(pid)
    #     if not p:
    #         continue
        
    # result.append({
    #     "product_id": pid,
    #     "name": p.name,
    #     "section": p.section,
    #     "price": Decimal(p.price),
    #     "quantity": p.quantity,
    #     "unit": p.unit,
    #     "similarity": round(score, 4),
    # })
    
    
    # return result

def find_using_synonyms(text, conn):
    cursor = conn.cursor()
    
    text_cleaned = remove_accents(text).lower()
    
    # Splitting the search text into words and adding wildcards
    words = [word.strip() for word in text_cleaned.split() if word.strip()]
    
    if not words:
        return []
    
    # Generating conditions for each word (case-insensitive)
    conditions = " AND ".join(["elem ILIKE %s" for _ in words])
    params = [f"%{word}%" for word in words]
    
    cursor.execute(f"""
                    SELECT name 
                    FROM products_with_embeddings 
                    WHERE EXISTS ( 
                        SELECT 1
                        FROM jsonb_array_elements_text(COALESCE(synonymous, '[]'::jsonb)
                        ) AS elem
                        WHERE {conditions}
                    )
                    LIMIT 30;
                   """, params)
    
    search_term = f"%{text}%"
    cursor.execute("""
                SELECT name, synonymous FROM products_with_embeddings 
                WHERE EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(COALESCE(synonymous, '[]'::jsonb) 
                    ) AS elem
                    WHERE elem LIKE %s
                )
                LIMIT 30;
                """, (search_term,))
    return cursor.fetchall()

# Test cases
# print(find_section("I want the product from seção B"))  
# print(find_section("Where's the seção A?"))            
# print(find_section("Can I see the seção 2?"))          
# print(find_section("seção There's no here")) 

# print(find_price("I want to see the price of all 2.50")) 
# print(find_price("I want to 3.50 price of all")) 
# print(find_price("I 100.20 to price of all")) 
# print(find_price("I 10020 to price of all")) 

