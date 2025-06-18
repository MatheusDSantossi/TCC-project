from utils.project_ecommerce_AI import (
    generate_synonymous as _gen_syn,
    fetch_products_alike as _fetch_alike,
    pre_process_prompt_grammatical,
    genai
)
from utils.prompt_searcher import find_section
from config.database_connection import conn
from utils.embeddings import embed_fn

def generate_synonyms_for(product_list=None, brand=None, synonyms=None, product=None, business=None):
    return _gen_syn(product_list=product_list, brand=brand, synonyms=synonyms, business=business) if product_list else _gen_syn(product=product, brand=brand, synonyms=synonyms, business=business)

def find_similar_products(name):
    return _fetch_alike(name)

def generate_embedding(name):
    return embed_fn(genai=genai, conn=conn, title=name, find_section=find_section)

def normalize_text(text):
    return pre_process_prompt_grammatical(text)
