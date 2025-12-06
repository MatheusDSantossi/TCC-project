import unicodedata
import re
from decimal import Decimal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import logging
from typing import List, Optional, Dict

from django.db import transaction
from django.db.models import Q
from django.contrib.postgres.search import TrigramSimilarity
from django.db.models.functions import Greatest
from django.db.models import F

from chatbot.models import Product
from utils.brands_utils import find_brands_in_text
from utils.metrics_utils import extract_metric_and_quantity

import google.generativeai as genai

import spacy

nlp = spacy.load("pt_core_news_sm")

logger = logging.getLogger(__name__)


def remove_accents(input_str):
    """Remove accents/diacritics from a string while preserving the base chracter"""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


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
    normalized = {sec.strip() for sec in raw_sections if sec and sec.strip()}

    # Sort case-insensitively
    sections = sorted(normalized, key=str.lower)

    logger.debug("Found sections (%d): %r", len(sections), sections)

    return sections


def find_section(text: str, business_id: Optional[int] = None):
    # Define the regular expression to match "section" followed by a letter or a number
    pattern = r"\bseção\s([a-zA-Z0-9]+)\b"

    # Search for the pattern in the input text (case-insensitive)
    match = re.search(pattern, text, re.IGNORECASE)

    # print("section Text (find section): ", text)

    # if a match is found, return the matched section
    if match:

        section = match.group(1)

        section = "Seção " + section

        print("Section after normalize: ", section)

        qs = Product.objects.filter(section=section)

        if business_id is not None:
            qs = qs.filter(business_id=business_id)

        results = qs.values("name", "section")[:30]

        print("List of products (find section): ", list(results))

        return list(results)

    return None


def find_price(text, business_id: Optional[int] = None):
    # Define the regular expression to match a price in the form of $99.99
    pattern = r"\b\d{1,3}(?:[.,]\d{2})\b"

    # Search for the pattern in the input text
    match = re.search(pattern, text)

    # +- 5 cents of difference in the price
    delta = Decimal("0.05")

    # if a match is found, return the matched price
    if match:
        price = match.group()
        price_with_point = Decimal(price.replace(",", "."))

        qs = Product.objects.filter(
            price__gte=price_with_point - delta, price__lte=price_with_point + delta
        )

        if business_id is not None:
            qs = qs.filter(business_id=business_id)

        results = qs.values_list("name", "price")[:30]

        print("Products find with the same price: ", list(results))

        return list(results)

    return None


def find_brand(text: str, business_id: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Look for known brand names in `text`. If found, return up to 30 matching products (name and brand). Returns None if no brand match.
    """

    # print("Text (find_brand): ", text)

    normalize_text = remove_accents(text).upper()

    # Normalize & detect brands
    brands = find_brands_in_text(normalize_text)

    if not brands:
        return None

    brand = brands[0]  # use the first match
    logger.debug("Brand match: %s in text: %r", brands, text)

    # Query via ORM
    qs = Product.objects.filter(brand__iexact=brand)

    if business_id is not None:
        qs = qs.filter(business_id=business_id)

    qs = qs.values("name", "brand")[:30]

    print("List of products (find brand): ", list(qs))

    return list(qs) or None


def find_quantity(text: str, business_id: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Extract quantity/unit information from text and find matching products.
    Returns a list of product dictionaries or None if no matches found.
    """
    # Preprocess text
    normalized_text = remove_accents(text).upper()
    logger.debug(f"Processing quantity for text: {normalized_text}")
    print(f"Processing quantity for text: {normalized_text}")

    # Extract quantity and metric using regex
    metric, quantity = None, None

    # Regex info:
    #   - an "x" with another number and then some letters (e.g., "1 X 24 500G")
    #   - or a standalone number with a unit (e.g., "500G", "500ML", "500MT")
    quantity_pattern = (
        r"\b\d+\s*(?:[Xx]\s*\d+\s*[A-Za-z]+|\s?(?:KG|G|ML|L|MT|CX|PT|PCT|UN|LTS|GR))\b"
    )

    match = re.search(quantity_pattern, normalized_text)

    if not match:
        logger.debug("No quantity/unity pattern found")
        print("No quantity/unity pattern found")
        return None

    quantity_str = match.group().strip()

    # Split of trailing letters as metric
    metric_match = re.search(r"[A-Za-z]+$", quantity_str)

    metric = metric_match.group().upper() if metric_match else None
    quantity = quantity_str

    logger.debug("Extracted quantity=%r, metric=%r", quantity, metric)
    print(f"Extracted quantity={quantity}, and metric={metric}")

    # Build ORM query
    query = Q()

    if quantity:
        query |= Q(quantity__iexact=quantity)

    if metric:
        query |= Q(unit__iexact=metric)

    if not query:
        logger.debug("No valid quantity or metric found")

        return None

    # Execute query
    qs = Product.objects.filter(query)

    if business_id:
        qs = qs.filter(business_id=business_id)

    # print(f"Found before name filter {len(qs)} matching products")

    # TODO: Think about a efficient way to narrow by product name. Considering that the text contains more than the product name.
    name_filter = normalized_text.replace(quantity_str, "").strip()

    # if name_filter:
    #     qs = qs.filter(name__icontains=name_filter)

    results = qs.values("name", "quantity", "unit")[:30]

    if not results:
        logger.debug("No matching products found")
        # print("No matching products found")
        return None

    logger.debug(f"Found {len(results)} matching products")
    # print(f"Found {len(results)} matching products")
    return list(results)


def extract_only_product_name(
    text: str, generate_content_model: genai.GenerativeModel
) -> str:
    # response = generate_content_model.generate_content(f"""
    #                                         **Improved Prompt: Product Extractor**

    #                                         You are a text corrector and product extractor. Your job is to read the user’s message, correct any minor typos, and return **only** the product names mentioned—nothing else. If no product appears, return **None**.

    #                                         Text to process:
    #                                         {text}

    #                                         **Processing Rules:**

    #                                         1. **Normalize** the text: fix spelling/grammar so it’s clear (e.g. “ssabonete” → “sabonete”).
    #                                         2. **Extract** all product names (common retail items) in the order they appear.
    #                                         3. **Return** a comma‑separated list of products, or the literal **None** if no products are found.
    #                                         4. Do **not** return any additional words, punctuation, formatting, or explanations.

    #                                         **Examples:**

    #                                         * Input: “Eu quero um biscoito de chocolate” → Output: biscoito de chocolate
    #                                         * Input: “Vocês têm desodorante?” → Output: desodorante
    #                                         * Input: “Quais produtos vocês têm?” → Output: None
    #                                         * Input: “Quanto custa o ssabonete?” → Output: sabonete
    #                                         * Input: “Como você está?” → Output: None
    #                                         * Input: “Este valor está correto?” → Output: None
    #                                         * Input: “Em qual seção fica o café e as bolachas?” → Output: café, bolachas

    #                                         Always assume the user is Brazilian—product names should stay in Portuguese.

    #                                         """)
    response = generate_content_model.generate_content(
        f"""
                **Improved Prompt: Product & Abbreviation Extractor**

                You are a text corrector and product extractor. Your job is to:
                1. Read the user’s message.
                2. Correct any minor typos.
                3. Extract **only** the product names mentioned.
                4. Generate all common abbreviations or shortened forms for each extracted product.
                5. Return the original product term followed by its abbreviations, separated by spaces.
                6. If no product appears, return **None**.

                Text to process:
                {text}

                **Processing Rules:**
                1. **Normalize** the text: fix spelling/grammar so it’s clear (e.g. “ssabonete” → “sabonete”).
                2. **Extract** all product names (common retail items) in the order they appear.
                3. For each product name, **generate** its common abbreviations or truncated forms (e.g., “shampoo” → “sh”, “shamp”, “shampoo”, “sham”).
                4. **Return** for each product a single line with the original name followed by its abbreviations, space‑separated.
                5. If no products are found, return **None**.
                6. Do **not** return any extra words, punctuation, formatting, or explanations.

                **Examples:**
                * Input: “Eu quero um biscoito de chocolate”  
                Output: biscoito chocolate bisc choc biscch
                * Input: “Vocês têm desodorante?”  
                Output: desodorante deso deo des
                * Input: “Quais shampoos vocês vendem?”  
                Output: shampoos sh shamp shampoo sham
                * Input: “Quais produtos vocês têm?”  
                Output: None

                Always assume the user is Brazilian—product names and abbreviations should stay in Portuguese.
                """
    )

    return response.text


def extract_product_chunk(text: str) -> str:
    doc = nlp(text)

    # Get the largest noun chunk
    chunks = list(doc.noun_chunks)

    if not chunks:
        return text

    # Choose the chunk containing the largest non-stopword content here we just pick the longest chunk by token count
    return max(chunks, key=lambda c: len(c)).text


def find_by_trigram_tokens(
    prod_phase, pre_qs, business_id=None, threshold=0.3, limit=30
):
    """
    Pre-filter by trigram similarity on tokens from prod_phrase.
    Returns a QuerySet annotated with 'similarity' and ordered by it (desc).
    """

    # Tokenize phrase - split on spaces, remove tiny tokens
    tokens = [t.strip() for t in prod_phase.split() if len(t.strip()) >= 2]

    if not tokens:
        # nothing to search for, return empty queryset
        return pre_qs.none()

    # Annotate a TrigramSimilarity for each token
    sims = [TrigramSimilarity("name", tok) for tok in tokens]

    # if only one similarity expression, annotate directly with that expression
    
    if len(sims) == 1:
        qs = pre_qs.annotate(similarity=sims[0]).filter(similarity__gt=threshold).order_by("-similarity")[:limit]

    else:
        # Use Greatest to pick the highest per-row similarity
        qs = pre_qs.annotate(similarity=Greatest(*sims)).filter(similarity__gt=threshold).order_by("-similarity")[:limit]

    return qs

    # return list(qs.values_list("product_id", "embeddings"))
    # return list(qs.values_list("product_id", "name", "similarity"))


# TODO: My plan is to fix the user_text before sending to the comparation between it and the products. Using these method (spacy) isn't very reliable the return are too different


# def find_similar_products_using_input_embeddings(input_embed: List[float], threshold: float = 0.4, top_k: int = 20, business_id: Optional[int]= None) -> List[Dict]:
def find_similar_products_using_input_embeddings(
    user_text: str,
    embed_fn,
    generate_content_model,
    threshold: float = 0.3,
    top_k: int = 20,
    business_id: Optional[int] = None,
) -> List[Dict]:
    """
    Given an input embedding, find the top_k most-similar products (similarity > threshold) and return their details + score.
    """

    # Extract the likely product phrase
    # prod_phrase = extract_product_chunk(user_text)
    prod_phrase = extract_only_product_name(user_text, generate_content_model)

    # print("prod_phrase: ", prod_phrase)

    # Hybrid pre-filter
    pre_qs = Product.objects.all()

    if business_id is not None:
        pre_qs = pre_qs.filter(business_id=business_id)

    # pre_qs = pre_qs.filter(name__icontains=prod_phrase)[:200]
    # pre_qs = pre_qs.annotate(similarity=TrigramSimilarity("name", prod_phrase)).filter(similarity__gt=threshold).order_by("-similarity")[:30]

    if prod_phrase.strip() == "None":
        return []

    pre_qs = find_by_trigram_tokens(prod_phrase, pre_qs, business_id, threshold)

    # rows = list(pre_qs.values_list("product_id", "embeddings"))
    rows = list(pre_qs.values_list("product_id", "embeddings"))

    # print("ROws: ", [ids[0] for ids in rows])
    print("ROws: ", rows)

    if not rows:
        return []

    # TODO: Enhance now from here down, because the products get are really good but the return or the continuation are losing them. Actually see above if we can improve as well

    product_ids, embeddings = zip(*rows)
    # print("products ids: ", product_ids)
    # print("embeddings: ", embeddings)
    embeddings = [np.array(e, dtype=np.float32) for e in embeddings]

    # Build input vector using embed_fn.
    #* IMPORTANT: adapt the call below to match your embed_fn signature.
    # Common pattern: embed_fn(text, business_id) -> vector
    # Compute similarity scores
    try:
        input_vec_raw = embed_fn(find_section, genai, business_id, prod_phrase)
        
    except TypeError:
        # Try alternative ordering (fallback)
        input_vec_raw = embed_fn(find_section, genai, business_id, prod_phrase)
    
    input_vec = np.array(input_vec_raw, dtype=np.float32).reshape(1, -1)
    
    sims = cosine_similarity(input_vec, embeddings)[0]

    # Pick & fetch details
    similar = sorted(
        [
            (product_ids[i], float(sims[i]))
            for i in range(len(product_ids))
            if sims[i] > threshold
        ],
        key=lambda x: -x[1],
    )[:top_k]

    if not similar:
        return []

    products = Product.objects.in_bulk(
        [pid for pid, _ in similar], field_name="product_id"
    )

    result = [
        {
            "product_id": pid,
            "name": prod.name,
            "section": prod.section,
            "price": prod.price,
            "quantity": prod.quantity,
            "unit": prod.unit,
            # "similarity": round(score,4),
        }
        for pid, score in similar
        if (prod := products.get(pid))
    ]

    # print("Find similar products using input embedding result: ", result)

    return result


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

    cursor.execute(
        f"""
                    SELECT name 
                    FROM products_with_embeddings 
                    WHERE EXISTS ( 
                        SELECT 1
                        FROM jsonb_array_elements_text(COALESCE(synonymous, '[]'::jsonb)
                        ) AS elem
                        WHERE {conditions}
                    )
                    LIMIT 30;
                   """,
        params,
    )

    search_term = f"%{text}%"
    cursor.execute(
        """
                SELECT name, synonymous FROM products_with_embeddings 
                WHERE EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(COALESCE(synonymous, '[]'::jsonb) 
                    ) AS elem
                    WHERE elem LIKE %s
                )
                LIMIT 30;
                """,
        (search_term,),
    )
    return cursor.fetchall()


# Test cases

# print(find_similar_products_using_input_embeddings("Quero um biscoito", embed_fn))

# print(find_section("I want the product from seção B"))
# print(find_section("Where's the seção A?"))
# print(find_section("Can I see the seção 2?"))
# print(find_section("seção There's no here"))

# print(find_price("I want to see the price of all 2.50"))
# print(find_price("I want to 3.50 price of all"))
# print(find_price("I 100.20 to price of all"))
# print(find_price("I 10020 to price of all"))
