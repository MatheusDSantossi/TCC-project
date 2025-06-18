import google.generativeai as genai

from pathlib import Path
import sys
import os
from dotenv import load_dotenv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure the root directory is included
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unicodedata

from fuzzywuzzy import fuzz

from config.config import SECRET_KEY

from config.database_connection import conn

import re

import ast

import json

import time
from .prompt_searcher import *
from .prompts import *
from .embeddings import embed_fn
from loaders import load_business_details, get_cached_products_df

# Você: quero um refrige

# Hecate: 'Temos refrigerantes Guaraná Antárctica e Soda Limonada, em tamanhos tradicionais e mini, na Seção 3. 😉'

# retornar mais produtos quando for generico

# Configurando a API do Gemini
GOOGLE_API_KEY = SECRET_KEY
genai.configure(api_key=GOOGLE_API_KEY)


# Generate a text stream
# response = model.generate_content(prompt, stream=True)
generation_config = {
    "temperature": 0.4,
}
    # "candidate_count": 1,

main_chatbot_model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                                generation_config=generation_config)

chat = main_chatbot_model.start_chat(
    history=[
        {"role": "user", "parts": "Olá"},
        {"role": "model", "parts": "É ótimo lhe conhecer. Como posso lhe ajudar?"}
    ]    
)

generate_content_model = genai.GenerativeModel(model_name="gemini-2.0-flash", 
                                               generation_config=generation_config)

# model_2 = genai.GenerativeModel("gemini-1.0-pro",
#                                 generation_config=generation_config)


# nltk.data.clear_cache()

business_name = "Supermercado Santos"

# According to Greek mythology, Hecate, the goddess 
# of magic, crossroads, and the night, is often depicted 
# as a benevolent deity who helps lost people find 
# their way
chatbot_name = "Hecate"                 


# index_dict = 0

# index_dict_answer = 0

# dict_of_client_products = {}

pre_processed_prompt_products_list = []

# nlp = spacy.load("pt_core_news_lg")

# Check if the text is an array. 
pattern = r"\[(.*?)\]"
# pattern_for_syn = r"\{.*?\}"
pattern_for_syn = r"\{(.+?)\}"

contains_a_product = True
products_list_after_user_input = []

text_has_section = False
text_has_brand = False
sections_find_in_db = find_business_sections(conn) 

# model_text_generator = models.get("text-davinci-003")
# def load_products_and_business_detail(business):
#     sql = f"SELECT * FROM products_with_embeddings WHERE business_id = {business}"

#     cur = conn.cursor()

#     try:
#         cur.execute(sql)

#         rows = cur.fetchall()
#     except (Exception, psycopg2.DatabaseError) as error:
#         rows = []
        
#     products_list = []

#     for row in rows:
#         products = {
#             "Nome": row[1],
#             "Secao": row[2],
#             "PrecoVenda": float(row[3]),
#             "UnidadeMedida": row[4],
#             "Embeddings": row[5],
#             "Synonymous": row[6],
#             "Brand": row[7],
#             "Quantity": row[8],
#             "stock": row[9],
#         }
#         products_list.append(products)

#     products_df = pd.DataFrame(products_list)

#     print("PRODUCTS DF: ", products_df)

#     # List of available products (I can keep that in the DB as well)
#     products_available = [p["Nome"] for p in products_df.to_dict(orient='records') if p['stock'] > 0]

#     # search_term = "REF"

#     # matching_products = [name for name in products_available if search_term in name]

#     # # print("Products available: ", products_available)
#     # print("Matching products: ", matching_products)

#     sql_business_detail = f"SELECT * FROM chatbot_businessdetail WHERE business_id = {business}"

#     cur.execute(sql_business_detail)

#     try:
#         cur.execute(sql_business_detail)

#         rows_business_detail = cur.fetchall()
#     except (Exception, psycopg2.DatabaseError) as error:
#         rows_business_detail = []

#     business_detail_list = []

#     for row in rows_business_detail:
#         business_detail = {
#             "Key": row[1],
#             "Value": row[2],
#         }
#         business_detail_list.append(business_detail)
        
#     print("Business detail list: ", business_detail_list)

#     business_detail_text = ""

#     for business_detail in business_detail_list:    
#         # print("Business detail: ", business_detail)
#         title = business_detail["Key"]
#         description = business_detail["Value"]
#         business_detail_text += f"Information title - {title}\n"
#         business_detail_text += f"Information description - {description}\n"

def for_dividing_prompt_limit(business):
    products_list_2 = []
    
    products_df = get_cached_products_df(business, conn)
    
    for p in products_df.to_dict(orient='records'):
        # prompt += p['Nome']
        products_list_2.append(p['Nome'])
        
    max_prompt_size = 425
    
    for i in range(0, max_prompt_size):
        prompt+= products_list_2[i]

def pre_process_prompt_brand(prompt, brands_find):
    fix_prompt = pre_process_prompt_grammatical_brand_fix_prompt(prompt, business_name, brands_find)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)
    
    # time.sleep(3)
    return pre_processed_prompt.text

def pre_process_prompt_section(prompt):
    products_find_in_section = find_section(prompt, conn)
    input_embed = embed_fn(find_section, genai, conn, prompt)
    products_find_using_prompt = find_similar_products_using_input_embeddings(input_embed, conn, np, cosine_similarity)
    products_alike = fetch_products_alike(prompt)
    
    fix_prompt = products_find_in_section_fix_prompt(products_find_in_section, products_find_using_prompt, products_alike, prompt)


    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)
    
    # time.sleep(3)
    return pre_processed_prompt.text

def pre_process_prompt_grammatical(prompt):
    
    fix_prompt = pre_process_prompt_grammatical_fix_prompt(prompt, business_name)
    
    time.sleep(3)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)

    # Problema agora é identificar mais de um produto :)

    return pre_processed_prompt.text

def fetch_brands_alike(product_brand):
    print("ENTERED FETCH BRANDS!!")
    with conn.cursor() as cursor:
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        
        query = """
            SELECT DISTINCT brand FROM products_with_embeddings
            WHERE SIMILARITY(brand, %s) > 0.1
            LIMIT 30;
        """
        
        # for keyword in product_name:
        cursor.execute(query, (product_brand,))
        rows = cursor.fetchall()
        
        unique_brands = [row[0] for row in rows]
        print("Unique brands: ", unique_brands)
            
    return unique_brands

def fetch_products_alike(product_name):
    with conn.cursor() as cursor:
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        
        query = """
            SELECT name, section, price FROM products_with_embeddings
            WHERE SIMILARITY(name, %s) > 0.1
            LIMIT 30;
        """
        
        # print(type(product_name))
        
        product_dict = {}
        
        # for keyword in product_name:
        cursor.execute(query, (product_name,))
        products = cursor.fetchall()
        
            
        for product in products:
            if product[0] not in product_dict:
                product_dict[product[0]] = []
            product_dict[product[0]].extend(product[1].split(","))
            
    return product_dict

def pre_process_prompt(prompt):
    
    global text_has_section 
    global text_has_brand 
    # fix_prompt_v1 = pre_process_prompt_grammatical(prompt)
    # print(f"FIXED 1 {fix_prompt_v1}")
    fix_prompt_v1 = prompt
    print(f"PROMPT 1 {fix_prompt_v1}")
    
    products_alike = fetch_products_alike(fix_prompt_v1)

    # print("PRODUCTS ALIKE: ", products_alike)
    
    input_embed = embed_fn(find_section, genai, conn, fix_prompt_v1)
    
    # print("INPUT EMBED: ", input_embed)
    
    products_find_in_section = find_section(fix_prompt_v1, conn)
    products_find_with_price = find_price(fix_prompt_v1, conn)
    products_find_using_prompt = find_similar_products_using_input_embeddings(input_embed, conn, np, cosine_similarity)
    products_find_using_brand = find_brand(fix_prompt_v1, conn)
    products_find_using_metric_or_quantity = find_quantity(fix_prompt_v1, conn)
    products_find_using_synonyms = find_using_synonyms(fix_prompt_v1, conn)
    
    fix_prompt_v2 = fix_prompt_v1
    
    if (products_find_in_section is not None 
    and products_find_with_price is not None 
    and products_find_using_brand is not None 
    and products_find_using_metric_or_quantity is not None):
        text_has_section = True
        text_has_brand = True
        # Match all conditions
        fix_prompt_v2 = products_match_all_conditions(
            products_find_in_section, 
            products_find_with_price, 
            products_find_using_brand, 
            products_find_using_metric_or_quantity, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    elif (products_find_in_section is not None 
        and products_find_with_price is not None 
        and products_find_using_brand is not None):
        text_has_section = True
        text_has_brand = True
        # Match section, price, and brand
        fix_prompt_v2 = products_match_section_price_and_brand(
            products_find_in_section, 
            products_find_with_price, 
            products_find_using_brand, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    elif (products_find_in_section is not None 
        and products_find_with_price is not None):
        text_has_section = True
        # Match section and price
        fix_prompt_v2 = products_match_section_and_price(
            products_find_in_section, 
            products_find_with_price, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )
    elif (products_find_in_section is not None and products_find_using_brand is not None):
        text_has_brand = True
        text_has_section = True
        products_match_section_and_brand(products_find_in_section, products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1)

    elif (products_find_in_section is not None and products_find_using_metric_or_quantity is not None):
        text_has_section = True
        products_match_section_and_metric_or_quantity(products_find_in_section, products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1)


    elif (products_find_with_price is not None 
        and products_find_using_brand is not None 
        and products_find_using_metric_or_quantity is not None):
        text_has_brand = True
        
        # Match price, brand, and metric
        fix_prompt_v2 = products_match_price_brand_and_metric(
            products_find_with_price, 
            products_find_using_brand, 
            products_find_using_metric_or_quantity, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    elif (products_find_with_price is not None 
        and products_find_using_brand is not None):
        text_has_brand = True
        
        # Match price and brand
        fix_prompt_v2 = products_match_price_and_brand(
            products_find_with_price, 
            products_find_using_brand, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )
    
    elif (products_find_with_price is not None 
        and products_find_using_metric_or_quantity is not None):
        # Match price and brand
        fix_prompt_v2 = products_match_price_and_quantity_or_metric(products_find_with_price, products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1)

    elif products_find_in_section is not None:
        text_has_section = True
        
        # Only match section
        fix_prompt_v2 = products_find_in_section_fix_prompt(
            products_find_in_section, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    elif products_find_with_price is not None:
        # Only match price
        fix_prompt_v2 = products_find_with_price_fix_prompt(
            products_find_with_price, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    elif products_find_using_brand is not None:
        text_has_brand = True
        
        # Only match brand
        fix_prompt_v2 = products_find_with_brand_fix_prompt(
            products_find_using_brand, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    elif products_find_using_metric_or_quantity is not None:
        # Only match metric or quantity
        fix_prompt_v2 = products_find_with_metric_or_quantity_fix_prompt(
            products_find_using_metric_or_quantity, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )
    elif products_find_using_synonyms is not None and products_find_using_metric_or_quantity is None and products_find_using_brand is None and products_find_using_prompt is None and products_find_with_price is None:
        fix_prompt_v2 = products_find_with_synonyms(
            products_find_using_synonyms,
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )

    else:
        # Fallback if no conditions match
        fix_prompt_v2 = products_find_general_fix_prompt(
            products_alike, 
            products_find_using_prompt, 
            fix_prompt_v1
        )
    
    time.sleep(3) # a little sleep to avoid overflow of questions to Gemini API
    
    print("FIX_PROMP_V2: ", fix_prompt_v2)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt_v2)

    return pre_processed_prompt.text

# pre_process_prompt("quais produtos custam 2.30")

# print(pre_process_prompt("Quero um cafe que custa 12 reais que está na seção 1 qual o nome desse cafe?"))


def create_semantic_dictionary(user_input, business):
    """
    Creates a semantic dictionary based on embeddings.

    Args:
        df (DataFrame): DataFrame with the products

    Returns:
        A dictionary where the keys are the products and the values are similar products lists
    """
    products_df = get_cached_products_df(business, conn)
    semantic_dictonary = {}
    # "REFRIG. GUARANA ANTARCT PET"	"Seção 3"	6	"PC"	"22021000"	4.89	4.89	6	true

    embeddings_consulta = np.array(embed_fn(find_section, genai, conn, user_input)).reshape(1, -1) # Reshape to 2D array for cosine similarity

    # print(f"Embed of products: {embeddings_consulta}")

    for i, row in products_df.iterrows():

        product_embedding = np.array(row['embeddings']).reshape(1, -1) # Ensure the product embedding is in the right shape
        
        similarity = cosine_similarity(embeddings_consulta, product_embedding)[0][0]
        
        semantic_dictonary[row["name"]] = similarity

    return semantic_dictonary

# print("SEMANTIC DIC: ",create_semantic_dictionary("Quais os produtos da seção 1"))


def encontrar_produtos_similares(semantic_dictionary, threshold=0.5):
    """
    Encontra produtos similares à consulta do usuário com base em embeddings.

    Args:
        consulta: A consulta do usuário como texto.
        threshold: O limiar de similaridade (0.0 a 1.0).

    Returns:
        Uma lista de nomes de produtos similares.
    """

    # doc_consulta = nlp(consulta)
    # embeddings_consulta = doc_consulta.vector


    similar_products = [product for product, similarity in semantic_dictionary.items() if similarity > threshold]

    return similar_products


# Função para identificar produtos na consulta do usuário
def find_similarity_products(prompt, embeddings_products=None, business=None):
    products_df = get_cached_products_df(business, conn)
    # in teh answer I have to distinguish between the answers with products and the answerrs without products (Olá, quero saber onde fica a seção 2)

    # avatar 
    # prompt_for_detail = "Gerar uma descrição detalhada do produto [nome do produto] para um chatbot de e-commerce"
 
    # fix_response = model_2.generate_content(pre_process_prompt(prompt))

    global pre_processed_prompt_products_list
    global pattern
    global contains_a_product

    fix_response = pre_process_prompt(prompt)

    prompt = fix_response
    
    print("CHATBOT RETURN: ", prompt)
    
    new_prompt = prompt.lower()
    new_prompt_without_spaces = new_prompt.strip()
    
    products_df['name'] = products_df['name'].str.lower()
    products_df['name'] = products_df['name'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii'))
    
    # print("ULTIMO CHARACTER " + new_prompt.strip()[-1])

    match = re.search(pattern, new_prompt)

    if match:
        # extracting the correspondent group to the pattern
        product_string = match.group(1)

        pre_processed_prompt_products_list = [product.strip() for product in product_string.split(',')]

    print("PRE_PROCESSED_PROMPT_PRODUCT: ", pre_processed_prompt_products_list)
    # PRE_PROCESSED_PROMPT_PRODUCT:  ['DES.ALMA FLORES SPRAY PCT 1 X 12 90ML', 'DES.ALMA FLORES SPRAY C.BRANCAS CXA 1 X 12 90ML', 'DES.SENADOR SPRAY SEDUCTION CXA 1 X 12 90ML', 'DES.SENADOR SPRAY CLASSIC CXA 1 X 12 90ML']
    
    
    if new_prompt_without_spaces not in ['produto na', 'texto ra']:
        print("new_prompt: ", new_prompt_without_spaces)
        # generate_synonymous(pre_processed_prompt_products_list)

        contains_a_product = True

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        
        new_prompt_clean = preprocess_text(new_prompt_without_spaces)
        print("new prompt clean: ", new_prompt_clean)
        products_df['Cleaned_Name'] = products_df['name'].apply(preprocess_text)

        # Using TF-IDF to text comparation
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([new_prompt_clean] + products_df['Cleaned_Name'].tolist())  

        similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

        # Getting the products index with similarity above the threshold

        threshold = 0.3
        # threshold = 0.5
        similar_index = np.where(similarities > threshold)[0]
        sorted_index = similar_index[np.argsort(-similarities[similar_index])]
        # index = np.argsort(similarities)[0][::-1]

        similar_products = sorted_index.tolist()
        
        print("Similar products: ", similar_products)

        # return index[0]
        # return index[:5].tolist()
        # return index[:quantity_products].tolist()
        # print(f"INDEX LIST {index[:5].tolist()}")
        return similar_products

    contains_a_product = False
    return prompt


def generate_and_search(prompt, business):
    print("Business gen and ser: ", business)
    products_df = get_cached_products_df(business, conn)
    print("Products df generate and serach: ", products_df)
    similarities = find_similarity_products(prompt, np.stack(products_df["embeddings"]), business)

    if isinstance(similarities, list):  

        return products_df.iloc[similarities]

    return similarities

def create_menu_options_prompts():
    query_select = """
        SELECT menu_option_description FROM menu_options 
        WHERE prompts IS NULL
    """
    with conn.cursor() as cursor:
        cursor.execute(query_select)
        options = cursor.fetchall()
        
        for option in options:
            print("OPTIONS: ", option[0])
            menu_prompt = f"""
                Considerando que a seguinte pergunta foi feita por um cliente: "{option[0]}"

                Para criar o melhor prompt possível para essa pergunta, leve em consideração os seguintes pontos:

                1. **Contexto**: O cliente está fazendo uma pergunta relacionada ao nosso supermercado. O objetivo é fornecer uma resposta precisa, direta e informativa, com base nos produtos e serviços que oferecemos.
                
                2. **Clareza**: O prompt deve ser claro o suficiente para que a IA entenda a intenção por trás da pergunta e seja capaz de gerar uma resposta que atenda às expectativas do cliente. Evite ambiguidades e garanta que a pergunta seja compreensível.
                
                3. **Personalização**: Sempre que possível, personalize o prompt para refletir a experiência do cliente em nosso supermercado. Isso pode incluir informações sobre produtos disponíveis, ofertas, recomendações personalizadas ou detalhes que mostrem atenção ao cliente.
                
                4. **Estilo e Tom**: A resposta gerada deve ser amigável e profissional, mantendo um tom acolhedor que reflita o atendimento ao cliente que nosso supermercado oferece. 

                5. **Objetivo**: Certifique-se de que o prompt gerado guiará a IA a fornecer informações úteis e relevantes sobre os produtos ou serviços que o cliente busca.

                Exemplo de como gerar um prompt para essa pergunta:
                Pergunta do cliente: "{option[0]}"
                Prompt ideal: "Baseado na pergunta do cliente sobre "tópico relevante", forneça uma resposta clara que ajude a responder diretamente à pergunta feita e inclua recomendações relacionadas aos nossos produtos ou serviços que possam ser de interesse para o cliente."
                
                no lugar de tópico relevantes use as informações abaixo que se adequarem mais a pergunta do cliente.

                \n
                Alguns produtos que vendemos que podem ser do interesse do cliente:\n
                {fetch_products_alike("refrigerante,suco,cafe")}
                
                \n
                
                Use a estrutura acima e ajuste para a pergunta dada.
                                                    """
            query_add = """
                UPDATE menu_options
                SET prompts = %s
                WHERE menu_option_description = %s
            """
            
            menu_prompt_generate = generate_content_model.generate_content(menu_prompt)
            response = menu_prompt_generate.text
            
            cursor.execute(query_add, (response, option[0]))
               
    
# user_input = "Onde fica o Refrigerante e Detergente e suco"

# user_input_processed = pre_process_prompt(user_input)

# print(user_input_processed)

# sematic_dict = create_semantic_dictionary(user_input_processed, products_df)

# # similarities = encontrar_produtos_similares(sematic_dict, 0.5)

# similarities = generate_and_search(user_input_processed)

# print(f"Produtos similares a '{similarities['Nome']}")

# um algoritmo que descreve a inclusão de um novo topico

def chatbot_response(user_prompt, text_related_to="default", business=None):
    
    global dict_of_client_products
    global products_list_after_user_input
    
    business_detail_list = load_business_details(business, conn)
    
    business_detail_text = "" 
    for k, v in business_detail_list.items():
        business_detail_text += k
        business_detail_text += ":\n"
        business_detail_text += v
        
    products_df = get_cached_products_df(business, conn)

    print("Products df: ", products_df)

    products_available = [p["name"] for p in products_df.to_dict(orient='records') if p['stock'] > 0]

    print("Text related to: ", text_related_to)

    if text_related_to == "brand":
        brands_find = fetch_brands_alike(user_prompt)
        
        brand = pre_process_prompt_brand(user_prompt, brands_find)
        fix_prompt_v1 = f"I want a product of this brand {user_prompt}, it has some products like these ones {brand}, which ones do you recommend?"
        
    elif text_related_to == "section":
        sections = pre_process_prompt_section(user_prompt)
        fix_prompt_v1 = f"I want a product of this section {user_prompt}, which have some produtcst like this {sections}, which ones do you recommend?"
    else:
        fix_prompt_v1 = pre_process_prompt_grammatical(user_prompt)
        
    find_product = generate_and_search(fix_prompt_v1, business)

    is_product = isinstance(find_product, pd.DataFrame)
    
    # USER PROMPT:  seção 3
    # Fixed prompt v1:  I want a product of this section seção 3, which have some produtcst like this ['REFRESCO  PLUS LARANJA', 'AGUA SANITARIA', 'SABAO LIQ.  MULTIACAO DOYP', 'FRALDA DESC. XG CXA', 'LEITE UHT  DESNATADO', 'REFRIG. SODA LIMONADA  PET', 'BISC. CIPANDOCE', 'TOALHA PAPEL SOCIAL CLEAN 50 FLS FDO', 'ACETONA', 'MAC.  SPECIALE PARAFUSO', 'SARD COQ OLEO', 'LAMP. 15W220V  ELETR.3U UND', 'PROTEINA DE SOJA  ESCURA T.A..', 'ACHOC.  CHOC. UHT', 'AGUA SANITARIA  2lt', 'MARG. POTE PTE', 'ACUCAR DEMERARA  FDO', 'BOMBOM  SORTIDO UND', 'SABAO PDC  AZUL PERFUMADO  200G', 'SARDINHA  PQ OLEO']
    # , which ones do you recommend?
    # Find products:  Empty DataFrame
    # TODO: Check why is returning an empty DataFrame!!
    
    print("USER PROMPT: ", user_prompt)
    print("Fixed prompt v1: ", fix_prompt_v1)
    print("Find products: ", find_product)

    if is_product:
        
        # for column in find_product.columns:
        #     print(f"column: {column}")
        
        count_row = find_product.shape[0]
        print(count_row)
        # if count_row > 1:
        if contains_a_product:
            # for index, row in find_product.iteritems():
            for index, row in find_product.iterrows():
            # print(f"ID: {index}, Nome: {row['Nome']}, Secao: {row['Secao']}, stock: {row['stock']}")
                dict_of_client_products = {
                    "Name": row['name'],
                    "Brand": row["brand"],
                    "Section": row['section'],
                    "Price": row['price'],
                    "Available": row['stock'],
                    "Quantity": row['quantity'],
                    "Metric": row['unit'],
                }            
                products_list_after_user_input.append(dict_of_client_products)                
        
            prompt_intro = "Generate a detailed yet compact description for an e-commerce chatbot using the following product data:\n\n"
                
            # print("products_list_after_user_input: ", products_list_after_user_input)
                
            name_of_product = ""
            matching_products = []
            product_details = [
                f"Product Name: {product['Name']}\n"
                f"Product Brand: {product['Brand']}\n"
                f"Product Price: {product['Price']}\n"
                f"Stock: {product['Available']}\n"
                f"Quantity Format: {product['Quantity']}\n"
                f"Metric Format: {product['Metric']}\n"
                for product in products_list_after_user_input
            ]
            
            detailed_prompt = prompt_intro + "\n".join(product_details) 
            
            print("Name of product: ",name_of_product)
            if (len(name_of_product) > 0):
                first_letter_of_product = name_of_product[:4]
                
                print("Four first letters: ", first_letter_of_product)

                matching_products = [name for name in products_available if first_letter_of_product.upper() in name]
            else:
                matching_products = products_available[:20]
        # print("Matching products: ", matching_products)
        # Generate a text stream
        # By default, the model returns a response after 
        # completing the entire text generation process. 
        # You can achieve faster interactions by not 
        # waiting for the entire result, and instead 
        # use streaming to handle partial results.

        # The following example shows how to implement streaming 
        # using the streamGenerateContent method to 
        # generate text from a text-only input prompt.
        generated_detailed_prompt = generate_content_model.generate_content(detailed_prompt)
        
        # TODO: Check why the product when I search for section

        # if find_product['stock'].item():
        if find_product['stock'].any():
            answer_prompt = f"""Considering that the customer of our store **{business_name}** sent the following message:

            "{fix_prompt_v1}"

            And the possible products they are referring to are:\n            
            """
            # for product in pre_processed_prompt_products_list: 
            for product in products_list_after_user_input: 
                answer_prompt += f"- **Product Name:** {product['Name']}\n"
                answer_prompt += f"- **Product Brand:** {product['Brand']}\n"
                answer_prompt += f"  - **Price:** {product['Price']}\n"
                answer_prompt += f"  - **Section:** {product['Section']}\n"
                answer_prompt += f"  - **Availability:** {product['Available']} in stock\n\n"

            # {pre_processed_prompt_products_list}.
            answer_prompt += f"""
            Additionally, consider this detailed product description as a reference:

            {generated_detailed_prompt.text}

            Use this description as a **guide** to improve the response to the customer.  
            Respond with a **clear, concise, and helpful sentence** that incorporates relevant details about the product, its description, and the section where it is located.  
            Do not include **any unnecessary text, symbols, or additional explanations**—just a **straightforward and informative message** based on the provided data.
            """
            
        else:
            answer_prompt = f"""Considering that the customer of our store **{business_name}** sent the following message:\n 
            
            "{fix_prompt_v1}"\n
            
            and that their desired product is currently unavailable,
            return a simple and useful message to the customer.
            Additionally, consider this detailed product text as a reference:\n
            {generated_detailed_prompt.text} \n
            
            And enhanced your answer.
             """

    else:
        
        products_alike = fetch_products_alike(user_prompt)
        
        if user_prompt == "Saber mais sobre os produtos":
            answer_prompt = f"""
                The customer from our store **{business_name}** has sent the following message:

                "{user_prompt}"

                Here are some of the products we currently have available: 
                \n
                
                {products_alike} \n
                """
                        
            with conn.cursor() as cursor:
                
                sql_query = """
                SELECT name, section, price, stock FROM products_with_embeddings
                ORDER BY RANDOM() LIMIT 20;
                """
                
                cursor.execute(sql_query)
                result = cursor.fetchall()
                
                
            
            product_info = ""
            
            for product in result:
                for i in range(len(product)):
                    # print("RANDOM PRODUCT: ", product[i])
                    product_info += str(product[i]) + " " 
                
                # answer_prompt+=product_info.strip() + "\n"
            
            print("Product info: ", product_info)
            
            # print("Fixed prompt V1: ", fix_prompt_v1)                
                
            
            answer_prompt+="""
            Considering these products, which include the following details in order: product name, product section, selling price, and availability,  
            generate a **concise and engaging response** showcasing a few of these products to see if the customer is interested in any of them.  
            Provide the **best possible response** based on the available data.
            """
            
        elif user_prompt == "Saber mais sobre nosso mercado":
            # TODO: Improve this part
            answer_prompt = f"""
                The customer from our store **{business_name}** has sent the following message:

                "{user_prompt}"

                Give useful information to them\n
                """
   
        else:
            
            answer_prompt = f"""The customer from our store **{business_name}** wants to know:

            "{fix_prompt_v1}"

            Provide a **clear and helpful response** based on the query.  
            If applicable, suggest an action or event the customer might be interested in.  
            For example:  

            - **Customer:** "What time does the store open?"  
            - **Chatbot:** "Our store opens at **7 AM**, and today we have a **big promotion** happening!"  
                """
                
            answer_prompt += """But only return this kind of information if it's related to the question of the client."""
        

    if text_has_section:
        answer_prompt+=f"""
        The user's message **mentions something related to store sections**. The available sections in the store are:

        {sections_find_in_db}

        Take this into account when generating the response.
        \n 
        """
    if text_has_brand:
        answer_prompt+=f"""
        The user's message **includes a brand or a product name associated with a brand**.  
        Ensure that this is considered when generating the response.
        """
    

    section_products = find_section(fix_prompt_v1, conn)
    
    if section_products:
        answer_prompt+=f"""
        Also, take into account that these products:

        {section_products}

        Belong to the section mentioned by the customer.  
        **Generate a relevant response including some examples** from these products.  

        If the data above contains a message like "This section isn't available in our establishment,"  
        inform the customer about this issue **politely and professionally**.
        """
        
    answer_prompt+= f"""
        Additionally, consider the following information about the {business_name}:\n
    """
    
    answer_prompt+=business_detail_text
    
    answer_prompt+=f"""
    But only, and just only use this information if it's related to the question of the client.\n
    """
    
    answer_prompt+="""
            Based on this information, generate a **useful and engaging response** for the customer,  
            including a possible follow-up action or question they might ask.  

            Avoid any unnecessary comments. At the end of your response,  
            **ask if you can assist with anything else** or offer a **related follow-up question**.
            Remember do not EVER return a response in English, you are talking with a Brazilian client. 
            """

    
    response = chat.send_message(answer_prompt)

    print("-"*80)
    print(f"ANSWER PROMPT {answer_prompt}")
    
    print("-"*80)
    print("RESPONSE: ", response.text)

    # return f"'{response.text}'"
    return response.text


# synonym_test = "biscoito cream cracker".capitalize()

# print(chatbot_response("Estou interessado na Seção 3"))

# print("Products using synonyms: ", find_using_synonyms(conn, synonym_test))
# print("Products using brand: ", find_brand( "Quero saber mais sobre a marca VITAMASSA", conn))

# print(chatbot_response("O produto que eu quero custa 2.50"))
# print(chatbot_response("O produto da marca even tem quais?"))
# print(chatbot_response("Quais produtos de 1KG vocês têm?"))
# print(chatbot_response("Quero um produto de 500G"))
# print(find_quantity("macarrao pilar 26X600G", conn))
def generate_synonymous(product=None, product_list=None, brand=None, synonyms=None, business=None):
    # TODO: Fix this structure because brand can't be None
    products_dict = {}
    product_synonymous_list = []
    
    if product_list:
        print("product list: ", product_list)
    
        print("product list name: ", product_list.name)
        original_product = product_list.name
        
        # semantic_dictorinary = create_semantic_dictionary(product)
        if brand:
            products = generate_and_search(brand, business)
        else:
            products = generate_and_search(original_product, business)
        
        if not isinstance(products, str):
            # print("products value: ", products)
            # print("Instance of product 1.0: ", isinstance(products, str))
            if products.size > 1:
                print("PRODUCTS SIZE: ", products.size)
                for index, row in products.iterrows():

                    products_dict = {
                        "Name": row['name'],
                        "Section": row['section'],
                        "Price": row['price'],
                        "Available": row['stock'],
                    }            
                    product_synonymous_list.append(products_dict)
                    # print("GENERATE SYNONYMOUS PRODUCT", products_dict["Name"])
    
    else:
        print("product: ", product)
        
        print("product name: ", product)
        original_product = product
        
        # semantic_dictorinary = create_semantic_dictionary(product)
        if brand:
            products = generate_and_search(brand, business)
        else:
            products = generate_and_search(original_product, business)
        
        
    if not isinstance(products, str):
        # print("products value: ", products)
        # print("Instance of product 2: ", isinstance(products, str))
        if products.size > 1:
            for index, row in products.iterrows():

                products_dict = {
                    "Name": row['name'],
                    "Section": row['section'],
                    "Price": row['price'],
                    "Available": row['stock'],
                }            
                product_synonymous_list.append(products_dict)
                # print("GENERATE SYNONYMOUS PRODUCT", products_dict["Name"])
    
    print("Products synonymous list: ", product_synonymous_list)
    print("Products dict list: ", products_dict)

    # print("""PRODUCTS: """, products)
    
    # ```python
    # {'refrig. guarana pet': ['Refrigerante Guaraná PET', 'Refrigerante Guaraná em Garrafa PET', 'Bebida Guaraná PET'], 'refrig. guarana pet': ['Refrigerante Guaraná PET', 'Refrigerante Guaraná em Garrafa PET', 'Bebida Guaraná PET'], 'refrig. guarana': ['Refrigerante Guaraná', 'Bebida Guaraná', 'Suco de Guaraná']}
    # ```
    # TODO: Fix the prompts because Gemini is returning some weird characters.
    
    if not products_dict:
    
        new_product = pre_process_prompt_grammatical(original_product)
        
        print("New product: ", new_product)
        prompt = """Generate all possible synonyms for the product names listed below. Follow these rules carefully:

        The product names may be abbreviated, e.g., "INSET SBP AERO" means "INSETICIDA SBP AEROSSOL". Analyze each abbreviation to determine its meaning.
        
        Assume the products are common retail items found in Brazilian supermarkets. Use this context to identify abbreviations and generate synonyms.
        
        Do not repeat synonyms, and limit the list to a maximum of 30 synonyms per product.
        
        Return the output strictly as a Python dictionary in the following format:
        {"NOME_DO_PRODUTO": ["sinônimo1", "sinônimo2", ...], "OUTRO_PRODUTO": ["sinônimo1", "sinônimo2", ...]}
        
        The synonyms must be in Brazilian Portuguese, even though this prompt is in English.
        
        Do not include any unnecessary text, comments, or formatting such as python, json, markdown, or anything similar. Only return the dictionary.
        
        Ensure all synonyms are enclosed in double quotes (e.g., "Caixa de Fósforos", "Palito de Fósforo").
        
        Here are the product names:
        \n"""
        
        prompt+=new_product
        
        prompt += """\n\nOutput example (adjusted for dynamic products):
        
        {"bisc. rosq. chocolate": ["biscuit chocolate ring", "cookie chocolate ring", "ring-shaped chocolate biscuit"], "bisc. rosq. leite": ["milk biscuit ring", "milk cookie ring", "ring-shaped milk biscuit"]}
        
        Important Notes:

        1. Do not include duplicate synonyms across or within products.
        
        2. Do not return any extra characters, comments, or text (e.g., no json, python, or similar formatting).
        
        3. The product list may change dynamically. Always adapt to the provided product names.
        
        4. ONLY return the synonymous don't return any other text, for example ```python ```json or anything like that, I want the response to start with a { and end with a }, the format that I've just told you about.
        """ 
    
    elif product_list: 
        print("Original product elid: ", original_product)
        new_product = pre_process_prompt_grammatical(original_product)
        
        print("New_product: ", new_product)
        
        prompt = """Generate all possible synonyms for the product names listed below. Follow these rules carefully:

        The product names may be abbreviated, e.g., "INSET SBP AERO" means "INSETICIDA SBP AEROSSOL". Analyze each abbreviation to determine its meaning.
        
        Assume the products are common retail items found in Brazilian supermarkets. Use this context to identify abbreviations and generate synonyms.
        
        Do not repeat synonyms, and limit the list to a maximum of 30 synonyms per product.
        
        Return the output strictly as a Python dictionary in the following format:
        {"NOME_DO_PRODUTO": ["sinônimo1", "sinônimo2", ...], "OUTRO_PRODUTO": ["sinônimo1", "sinônimo2", ...]}
        
        The synonyms must be in Brazilian Portuguese, even though this prompt is in English.
        
        Do not include any unnecessary text, comments, or formatting such as python, json, markdown, or anything similar. Only return the dictionary.
        
        Ensure all synonyms are enclosed in double quotes (e.g., "Caixa de Fósforos", "Palito de Fósforo").
        
        Here are the product names:
        \n"""
        
        prompt+=new_product
        
        prompt += """\n\nOutput example (adjusted for dynamic products):
        
        {"bisc. rosq. chocolate": ["biscuit chocolate ring", "cookie chocolate ring", "ring-shaped chocolate biscuit"], "bisc. rosq. leite": ["milk biscuit ring", "milk cookie ring", "ring-shaped milk biscuit"]}
        
        Important Notes:

        1. Do not include duplicate synonyms across or within products.
        
        2. Do not return any extra characters, comments, or text (e.g., no json, python, or similar formatting).
        
        3. The product list may change dynamically. Always adapt to the provided product names.
        
        4. ONLY return the synonymous don't return any other text, for example ```python ```json or anything like that, I want the response to start with a { and end with a }, the format that I've just told you about.
        """ 
     
    else:
        print("Product name: else ", product)
        
        prompt = """Generate all possible synonyms for the product names listed below. Follow these rules carefully:

        The product names may be abbreviated, e.g., "INSET SBP AERO" means "INSETICIDA SBP AEROSSOL". Analyze each abbreviation to determine its meaning.
        
        Assume the products are common retail items found in Brazilian supermarkets. Use this context to identify abbreviations and generate synonyms.
        
        Do not repeat synonyms, and limit the list to a maximum of 30 synonyms per product.
        
        Return the output strictly as a Python dictionary in the following format:
        {"NOME_DO_PRODUTO": ["sinônimo1", "sinônimo2", ...], "OUTRO_PRODUTO": ["sinônimo1", "sinônimo2", ...]}
        
        The synonyms must be in Brazilian Portuguese, even though this prompt is in English.
        
        Do not include any unnecessary text, comments, or formatting such as python, json, markdown, or anything similar. Only return the dictionary.
        
        Ensure all synonyms are enclosed in double quotes (e.g., "Caixa de Fósforos", "Palito de Fósforo").
        
        Here are the product names:
                \n"""
        
        prompt += product
        
        # for product in product_synonymous_list:
        #     prompt += str(product["Name"]) + "\n"
        
        prompt += """\n\nOutput example (adjusted for dynamic products):
        
        {"bisc. rosq. chocolate": ["biscuit chocolate ring", "cookie chocolate ring", "ring-shaped chocolate biscuit"], "bisc. rosq. leite": ["milk biscuit ring", "milk cookie ring", "ring-shaped milk biscuit"]}
        
        Important Notes:

        1. Do not include duplicate synonyms across or within products.
        
        2. Do not return any extra characters, comments, or text (e.g., no json, python, or similar formatting).
        
        3. The product list may change dynamically. Always adapt to the provided product names.
        
        4. ONLY return the synonymous don't return any other text, for example ```python ```json or anything like that, I want the response to start with a { and end with a }, the format that I've just told you about.
        """ 

    if (synonyms):
        prompt+= f"consider these are some examples of synonyms for the product that the employee choose: {synonyms} \n"
        
        prompt+= f"add these synonyms in your final answer and take them as example to return all synonyms."
    
    print("GENERATE SYNONYMOUS PROMPT: ", prompt)
    print("==========================================")
    
    # print("PROMPT:", prompt)   
    response = generate_content_model.generate_content(prompt)
    print(response.text)
    
    # {'Macarrão Pilsen Médio': ['Macarrão', 'Massa', 
    #                            'Espaguete', 'Talharim', 'Fideli', 'Ninhos', 'Penne', 'Farfalle', 'Spaghetti', 'Macarrão Tipo Pilsen', 'Macarrão Médio', 'Macarrão de Trigo', 'Macarrão Seco', 'Macarrão Cozido', 'Macarrão para Sopa', 'Macarrão para Salada', 'Macarrão para Massa']}
    
    cleaned_response = re.sub(r"^```[a-zA-Z]*\n|```$", "", response.text).strip()
    
    if not cleaned_response:
         cleaned_response = re.sub(r"^```[a-zA-Z]*\n", "", response.text).strip()
    
    print(cleaned_response)
    
    try:

        synonymous_dict = ast.literal_eval(cleaned_response)
        
        first_key = next(iter(synonymous_dict))
        
        synonymous_list = synonymous_dict[first_key]
        
        synonymous_list = json.dumps(synonymous_list, ensure_ascii=False)
        
        print("Synonymous list: ", synonymous_list)
        
        sanitized_synonymous_list = str(synonymous_list).replace('[', '{').replace(']', '}')
        
        print("Sanitized synonyms:", sanitized_synonymous_list)
 
        if isinstance(product, dict):
            print("product: ", product['Name'])
            product = product['Name']

        original_product = original_product.strip()

        print("original product: ", original_product)
        with conn.cursor() as cursor:
            cursor.execute("""
                        UPDATE products_with_embeddings
                        SET synonymous = %s
                        WHERE name = %s
                        """, (synonymous_list, original_product))
            
                
        return synonymous_list
    
    except Exception as e:
        print("Error generating synonyms: ", e)
        return None

# print(generate_synonymous("REFRIG. GUARANA  PET"))
