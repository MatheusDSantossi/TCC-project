import pandas as pd
from django.db import connection

from functools import lru_cache

def load_products_for_business(business_id):
    """
    Returns a pandas DataFrame of products for the given business.
    """
    print("Business_id: ", business_id)
    sql = """ 
        SELECT
            *
        FROM products_with_embeddings
        WHERE business_id = %s;
    """
    
    with connection.cursor() as cur:
        cur.execute(sql, (business_id,))
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        
    # Building the DataFrame
    df = pd.DataFrame(rows, columns=cols)
    
    return df
        
def load_business_details(business_id):
    sql = """
        SELECT key, value FROM chatbot_businessdetail
        WHERE business_id = %s
    """
    print("Business_id: ", business_id)
    
    with connection.cursor() as cur:
        cur.execute(sql, (business_id,))
        return dict(cur.fetchall())

@lru_cache(maxsize=10)
def get_cached_products_df(business_id):
    return load_products_for_business(business_id)
