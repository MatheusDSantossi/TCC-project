from utils.project_ecommerce_AI import (
    generate_synonymous as _gen_syn,
    fetch_products_alike as _fetch_alike,
    pre_process_prompt_grammatical,
)
from utils.embeddings import embed_fn

def generate_synonyms_for(product_list=None, brand=None, synonyms=None, product=None):
    return _gen_syn(product_list=product_list, brand=brand, synonyms=synonyms) if product_list else _gen_syn(product=product, brand=brand, synonyms=synonyms)

def find_similar_products(name):
    return _fetch_alike(name)

def generate_embedding(name):
    return embed_fn(name)

def normalize_text(text):
    return pre_process_prompt_grammatical(text)
