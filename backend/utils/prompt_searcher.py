import re
from changing_db_columns import KNOWN_BRANDS, KNOWN_METRICS

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
            # cursor.execute("SELECT name FROM products_with_embeddings WHERE secao_produto = %s", (match.group(1),))
            
            cursor.execute("""
                           SELECT DISTINCT secao_produto FROM products_with_embeddings
                           """)
            
            sections_in_database = cursor.fetchall()
            
            print("Sections in database: ", sections_in_database)
            print("Length of Sections in database: ", len(sections_in_database))
        
            if section in [s[0] for s in sections_in_database]:
                print("Section found in database")
            
                cursor.execute("""
                            SELECT name FROM products_with_embeddings 
                            WHERE secao_produto = %s
                            LIMIT 30
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
                           SELECT name FROM products_with_embeddings 
                           WHERE preco_venda = %s
                           LIMIT 30
                           """, (price_with_point,))
            return cursor.fetchall()
    
    return None

def find_brand(text, conn):
    
    brand = None
    
    # Define the regular expression to match "brand"
    # Check for known brands
    for known_brand in KNOWN_BRANDS:
        if known_brand.lower() in text.lower():
            brand = known_brand
            break
        
    if brand:
        with conn.cursor() as cursor:
            
            cursor.execute("""
                            SELECT name, brand FROM products_with_embeddings 
                            WHERE brand = %s
                            LIMIT 30
                            """, (brand,))
            
            return cursor.fetchall()
    
    return brand

def find_quantity(text, conn):

    metric = None
    quantity = None

    # Identify quantity (e.x., "6X30", "1X24 500G")
    quantity_match = re.search(r"\b\d+\s*[Xx]\s*\d+[A-Za-z]*\b|\b\d+\s?(KG|G|ML|L)\b", text, re.IGNORECASE)

    
    if quantity_match:
        quantity = quantity_match.group().strip()
        text = text.replace(quantity, '').strip()
    
    # Check for known metrics
    
    for known_metric in KNOWN_METRICS:
        if re.search(rf"\b{known_metric}", text, re.IGNORECASE):
            metric = known_metric
            text = re.sub(rf"\b{known_metric}\b", '', text, flags=re.IGNORECASE).strip()
            break
        
    if metric or quantity:
        
       with conn.cursor() as cursor:
            if metric and quantity:
                cursor.execute("""
                    SELECT name FROM products_with_embeddings
                    WHERE quantity = %s OR unidade_medida = %s
                    LIMIT 30
                """, (quantity, metric))
            elif quantity:
                cursor.execute("""
                    SELECT name FROM products_with_embeddings
                    WHERE quantity = %s
                    LIMIT 30
                """, (quantity,))
            elif metric:
                cursor.execute("""
                    SELECT name FROM products_with_embeddings
                    WHERE unidade_medida = %s
                    LIMIT 30
                """, (metric,))
            
            # Fetch and return the results
            results = cursor.fetchall()
            # return [row[0] for row in results]
            return results

    return None

def find_similar_products_using_input_embeddings(input_embed, conn, np, cosine_similarity):
    

    with conn.cursor() as cursor:
        
        # query = "CREATE EXTENSION IF NOT EXISTS vector"
        
        # cursor.execute(query)
        # conn.commit()
        
        query = """
            SELECT product_id, embeddings FROM products_with_embeddings
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
            cursor.execute("SELECT name, secao_produto, preco_venda FROM products_with_embeddings WHERE product_id = %s", (product_id,)
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

# Test cases
# print(find_section("I want the product from seção B"))  
# print(find_section("Where's the seção A?"))            
# print(find_section("Can I see the seção 2?"))          
# print(find_section("seção There's no here")) 

# print(find_price("I want to see the price of all 2.50")) 
# print(find_price("I want to 3.50 price of all")) 
# print(find_price("I 100.20 to price of all")) 
# print(find_price("I 10020 to price of all")) 
