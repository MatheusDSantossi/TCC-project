import re
# from chatbot_test import embeddings_testing

def find_section(text, conn):
    # Define the regular expression to match "section" followed by a letter or a number
    pattern = r'\bseção\s([a-zA-Z0-9]+)\b'
    
    # Search for the pattern in the input text (case-insensitive)
    match = re.search(pattern, text, re.IGNORECASE)
    
    
    # if a match is found, return the matched section
    if match:
        section = match.group(1)
        section_correct = "Seção " + section
        
        # return match.group(1)
        with conn.cursor() as cursor:
            # cursor.execute("SELECT nome FROM products_with_embeddings WHERE secao_produto = %s", (match.group(1),))
            cursor.execute("""
                           SELECT nome FROM products_with_embeddings 
                           WHERE secao_produto = %s
                           LIMIT 30
                           """, (section_correct,))
            return cursor.fetchall()
    
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
                           SELECT nome FROM products_with_embeddings 
                           WHERE preco_venda = %s
                           LIMIT 30
                           """, (price_with_point,))
            return cursor.fetchall()
    
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
        
        conn.commit()
        
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
            cursor.execute("SELECT nome, secao_produto, preco_venda FROM products_with_embeddings WHERE product_id = %s", (product_id,)
                           )
            product_row = cursor.fetchone()
            product_details.append({
                # "product_id": product_id,
                "nome": product_row[0],
                "preco_venda": product_row[1],
                "secao_produto": product_row[2],
                # "similarity": similarity
            })
            
        return product_details

# Test cases
# print(find_section("I want the product from seção B"))  
# print(find_section("Where's the seção A?"))            
# print(find_section("Can I see the seção 2?"))          
# print(find_section("seção There's no here")) 

# print(find_price("I want to see the price of all 2.50")) 
# print(find_price("I want to 3.50 price of all")) 
# print(find_price("I 100.20 to price of all")) 
# print(find_price("I 10020 to price of all")) 
