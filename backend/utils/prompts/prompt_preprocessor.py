from .prompt_preprocessing import pre_process_prompt
from typing import Optional
from ..embeddings import embed_fn

def preprocess_with_ai(
    prompt: str,
    find_section,
    find_price,
    find_brand,
    find_quantity,
    find_using_synonyms,
    find_similar_products_using_input_embeddings,
    fetch_brands_alike,
    generate_content_model,
    business: Optional[int] = None
) -> str:
    """
    Run the full AI-based prompt fixer pipeline and return a cleaned prompt that contains a comma-separated list of product names (if any).
    """
    
    return pre_process_prompt(
        prompt,
        find_section,
        find_price,
        find_brand,
        find_quantity,
        find_using_synonyms,
        find_similar_products_using_input_embeddings,
        embed_fn,
        fetch_brands_alike,
        generate_content_model,
        business
    )