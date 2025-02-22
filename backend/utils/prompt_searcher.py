import unicodedata
import re
from changing_db_columns import KNOWN_BRANDS, KNOWN_METRICS

def find_business_sections(conn):
    with conn.cursor() as cursor:

        cursor.execute("""
                        SELECT DISTINCT secao_produto FROM products_with_embeddings
                        """)
        
        sections_in_database = cursor.fetchall()
    return sections_in_database

def find_section(text, conn):
    # Define the regular expression to match "section" followed by a letter or a number
    pattern = r'\bseção\s([a-zA-Z0-9]+)\b'
    
    # Search for the pattern in the input text (case-insensitive)
    match = re.search(pattern, text, re.IGNORECASE)
    
    
    # if a match is found, return the matched section
    if match:
        section = match.group(1)
        
        section = "Seção " + section
        
        # return match.group(1)
        with conn.cursor() as cursor:
        #     # cursor.execute("SELECT name FROM products_with_embeddings WHERE secao_produto = %s", (match.group(1),))
            
        #     cursor.execute("""
        #                    SELECT DISTINCT secao_produto FROM products_with_embeddings
        #                    """)
            
        #     sections_in_database = cursor.fetchall()
            sections_in_database = find_business_sections(conn)
            
            print("Sections in database: ", sections_in_database)
            print("Length of Sections in database: ", len(sections_in_database))
        
            if section in [s[0] for s in sections_in_database]:
                print("Section found in database")
            
                cursor.execute("""
                            SELECT name, secao_produto FROM products_with_embeddings 
                            WHERE secao_produto = %s
                            LIMIT 20
                            """, (section,))
                return cursor.fetchall()
            return ["This section isn't available in our establishment"]
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
                           SELECT name, preco_venda FROM products_with_embeddings 
                           WHERE preco_venda = %s
                           LIMIT 20
                           """, (price_with_point,))
            return cursor.fetchall()
    
    return None

def remove_accents(input_str):
    """Remove accents/diacritics from a string while preserving the base chracter"""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def find_brand(text, conn):
    
    brand = None
    text_lower = text.lower()
    brand_clean = remove_accents(text_lower)
    
    # Creating a set of normalized brand names
    normalized_brands = {
        remove_accents(brand).lower(): brand for brand in KNOWN_BRANDS
        }
    
    # Tokenized input text
    tokens = re.findall(r'\b\w+\b', brand_clean)
    
    # Check for exact matches in tokens
    for token in tokens:
        # if known_brand.lower() in text.lower():
        if token in normalized_brands:
            brand = normalized_brands[token]
            print(f"Normalized brand match found: {brand}(Original: {text})")
            break
        
    if brand:
        print(f"Exact brand match found: {brand}")
        with conn.cursor() as cursor:
            
            cursor.execute("""
                            SELECT name, brand FROM products_with_embeddings 
                            WHERE brand = %s
                            LIMIT 30
                            """, (brand,))
            
            return cursor.fetchall()
    
    return None
    
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
                    SELECT name, quantity, unidade_medida FROM products_with_embeddings
                    WHERE quantity = %s OR unidade_medida = %s
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
                    SELECT name, unidade_medida FROM products_with_embeddings
                    WHERE unidade_medida = %s
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
            cursor.execute("SELECT name, secao_produto, preco_venda, quantity, unidade_medida FROM products_with_embeddings WHERE product_id = %s", (product_id,)
                           )
            product_row = cursor.fetchone()
            product_details.append({
                # "product_id": product_id,
                "nome": product_row[0],
                "preco_venda": product_row[1],
                "secao_produto": product_row[2],
                # "similarity": similarity
            })
            
        return product_details[:15]

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
                        SELECT *
                        FROM unnest(synonymous) AS elem
                        WHERE {conditions}
                    )
                    LIMIT 30;
                   """, params)
    
    search_term = f"%{text}%"
    cursor.execute("""
                SELECT name, synonymous FROM products_with_embeddings 
                WHERE EXISTS (
                    SELECT *
                    FROM unnest(synonymous) AS elem
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

