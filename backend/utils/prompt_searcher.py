import unicodedata
import re
import psycopg2

import logging
from typing import List, Optional, Dict

from django.db.models import Q

from chatbot.models import Product
from utils.brands_utils import find_brands_in_text
from utils.metrics_utils import extract_metric_and_quantity

logger = logging.getLogger(__name__)

# def find_business_sections(conn):
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
    
def find_section(text: str, conn, business_id: Optional[int] = None):
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
            
        qs = qs.values("name", "section")[:30]
    
        print("List of products (find section): ", list(qs))
        
        return list(qs)
    
        # return match.group(1)
        # with conn.cursor() as cursor:
        #     sections_in_database = find_business_sections()
            
        #     print("Sections in database: ", sections_in_database)
        #     print("Length of Sections in database: ", len(sections_in_database))
        
        #     if section in sections_in_database:
        #         print("Section found in database: ", section)
            
        #         cursor.execute("""
        #                     SELECT name, section FROM products_with_embeddings 
        #                     WHERE section = %s
        #                     LIMIT 20
        #                     """, (section,))
        #         return cursor.fetchall()
        #     return ["This section isn't available in our establishment"]
    return None

def find_price(text, conn):
    # Define the regular expression to match a price in the form of $99.99
    pattern = r'\b\d{1,3}(?:[.,]\d{2})\b'
    
    # Search for the pattern in the input text
    match = re.search(pattern, text)
    
    # if a match is found, return the matched price
    if match:
        price = match.group()
        price_with_point = price.replace(',', '.')
        # return float(match.group())
        with conn.cursor() as cursor:
            cursor.execute("""
                           SELECT name, price FROM products_with_embeddings 
                           WHERE price = %s
                           LIMIT 20
                           """, (price_with_point,))
            return cursor.fetchall()
    
    return None

def remove_accents(input_str):
    """Remove accents/diacritics from a string while preserving the base chracter"""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u''.join([c for c in nfkd_form if not unicodedata.combining(c)])

# def find_brand(text, conn):
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

    
    
    
def find_quantity(text, conn):

    metric = None
    quantity = None

    # Regex info:
    #   - an "x" with another number and then some letters (e.g., "1 X 24 500G")
    #   - or a standalone number with a unit (e.g., "500G", "500ML", "500MT")
    quantity_pattern = r"\b\d+\s*(?:[Xx]\s*\d+\s*[A-Za-z]+|\s?(?:KG|G|ML|L|MT|CX|PT|PCT|UN))\b"
    quantity_match = re.search(quantity_pattern, text, re.IGNORECASE)

    
    if quantity_match:
        quantity = quantity_match.group().strip()
        text = text.replace(quantity, '').strip()
        
        # Extract the trailing letters from the quantity as the metric.
        m = re.search(r"[A-Za-z]+$", quantity)
        
        if m:
            metric = m.group().upper()
    
    # Check for known metrics
    name = text.strip()
    print("Product name: ", name)
    print("Product metric: ", metric)
    print("Product quantity: ", quantity)
    if metric or quantity:
        
       with conn.cursor() as cursor:
            if metric and quantity:
                cursor.execute("""
                    SELECT name, quantity, unit FROM products_with_embeddings
                    WHERE quantity = %s OR unit = %s
                    LIMIT 20
                """, (quantity, metric))
            elif quantity:
                cursor.execute("""
                    SELECT name, quantity FROM products_with_embeddings
                    WHERE quantity = %s
                    LIMIT 20
                """, (quantity,))
            elif metric:
                cursor.execute("""
                    SELECT name, unit FROM products_with_embeddings
                    WHERE unit = %s
                    LIMIT 20
                """, (metric,))
            
            # Fetch and return the results
            results = cursor.fetchall()
            if len(results) == 0:
                return None
            # return [row[0] for row in results]
            return results

    return None

# TODO: Fix this function because the return is not good
def find_similar_products_using_input_embeddings(input_embed, conn, np, cosine_similarity):
    

    with conn.cursor() as cursor:
        
        # query = "CREATE EXTENSION IF NOT EXISTS vector"
        
        # cursor.execute(query)
        # conn.commit()
        
        query = """
            SELECT product_id, embeddings FROM products_with_embeddings
            LIMIT 10
        """
        
        cursor.execute(query)
        
        rows = cursor.fetchall()
        
        # conn.commit()
        
        # Convert the embeddings to a numpy arrays
        product_ids = [row[0] for row in rows]
        embeddings = [np.array(row[1], dtype=np.float32) for row in rows]
        
        # Convert the input embeddings to a numpy array
        input_embedding = np.array(input_embed, dtype=np.float32)
        
        # embeddings_testing(input_embedding, embeddings)
                
        # similarities = cosine_similarity([input_embed], embeddings)[0]
        similarities = cosine_similarity([input_embedding], embeddings)[0]
        
        # Find the most similar embeddings and their indices
        most_similar_indices = np.argsort(similarities)[::-1][:20]
        
        # Fetch the products_ids and similarities for the top similar products
        
        threshold = 0.5
        
        similar_products = [(product_ids[idx], similarities[idx]) for idx in most_similar_indices if similarities[idx] > threshold]
        # similar_products = [(product_ids[idx], similarities[idx]) for idx in most_similar_indices]
        
        product_details = []
        for product_id, similarity in similar_products:
            cursor.execute("SELECT name, section, price, quantity, unit FROM products_with_embeddings WHERE product_id = %s", (product_id,)
                           )
            product_row = cursor.fetchone()
            product_details.append({
                # "product_id": product_id,
                "nome": product_row[0],
                "price": product_row[1],
                "section": product_row[2],
                # "similarity": similarity
            })
            
        return product_details[:30]

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

