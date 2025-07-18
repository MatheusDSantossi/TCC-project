from django.db import connection

def fetch_brands_alike(product_brand: str, limit = 30, threshold: float = 0.1):
    """
    Return up to `limit` distinct brands whose name similarity to product_brand exceeds `threshold`, using pg_trgm.
    """
    with connection.cursor() as cursor:
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        
        query = """
            SELECT DISTINCT brand FROM products_with_embeddings
            WHERE SIMILARITY(brand, %s) > %s
            LIMIT %s;
        """
        
        cursor.execute(query, (product_brand, threshold, limit))
        
        return [row[0] for row in cursor.fetchall()]        
        
def fetch_products_alike(product_name: str, limit: int = 30, threshold: float = 0):
    """
    Return a dict mapping product_name -> [sections..., price] for products whose name-similarity to product_name exceeds threshold.
    """
    
    with connection.cursor() as cursor:
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        
        query = """
            SELECT name, section, price FROM products_with_embeddings
            WHERE SIMILARITY(name, %s) > %s
            LIMIT %s;
        """
                
        result = {}
        
        # for keyword in product_name:
        cursor.execute(query, (product_name, threshold, limit,))
        products = cursor.fetchall()
        
            
        for name, section, price in products:
            if name not in result:
                result[name] = []
            result[name].extend(section.split(","))
            result[name].append(price)
            
    return result
