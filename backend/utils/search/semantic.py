from django.db import connection
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.embeddings import embed_fn
from loaders import get_cached_products_df
import google.generativeai as genai

def create_semantic_dictionary(user_input: str, find_section, business) -> dict:
    """
    Creates a semantic dictionary based on embeddings.

    Args:
        df (DataFrame): DataFrame with the products

    Returns:
        A dictionary where the keys are the products and the values are similar products lists
    """
    products_df = get_cached_products_df(business)
    semantic_dictonary = {}
    # "REFRIG. GUARANA ANTARCT PET"	"Seção 3"	6	"PC"	"22021000"	4.89	4.89	6	true

    embeddings_consulta = np.array(embed_fn(find_section, genai, connection, user_input)).reshape(1, -1) # Reshape to 2D array for cosine similarity

    for _, row in products_df.iterrows():

        product_embedding = np.array(row['embeddings']).reshape(1, -1) # Ensure the product embedding is in the right shape
        
        similarity = cosine_similarity(embeddings_consulta, product_embedding)[0][0]
        
        semantic_dictonary[row["name"]] = similarity

    return semantic_dictonary

# print("SEMANTIC DIC: ",create_semantic_dictionary("Quais os produtos da seção 1"))


def encontrar_produtos_similares(semantic_dictionary: dict, threshold: float =0.5) -> list:
    """
    Return a list of product names whose similarity exceeds threshold.
    """

    similar_products = [product for product, similarity in semantic_dictionary.items() if similarity > threshold]

    return similar_products
