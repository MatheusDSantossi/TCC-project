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

import psycopg2
from config.database_connection import conn

import re

import ast

import json

import time
from prompt_searcher import *
from prompts import *

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

main_chatbot_model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b",
                                generation_config=generation_config)

chat = main_chatbot_model.start_chat(
    history=[
        {"role": "user", "parts": "Olá"},
        {"role": "model", "parts": "É ótimo lhe conhecer. Como posso lhe ajudar?"}
    ]    
)

generate_content_model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b", 
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

# My plan is to give the opportunity and way to the client put some information like the ones bellow
info_about_the_business = ["Nosso mercado abre as 7h e fecha as 22h", 
                           "Nosso mercado tem o valor de sempre atender os clientes com o melhor atendimento possível",
                           "O nome do nosso gerente é Matheus",
                           "As promoções do nosso mercado hoje são principalmente no arroz que está 50% mais barato",
                           f"""Essas são as categorias por tipo de produto do {business_name}:
                           Alimentos básicos, como arroz, feijão, cuscuz, macarrão, óleo, sal e açucar.
                           Frios e laticínios, como queijo qualho, manteigas e queijo de manteiga.
                           Bebidas, como refrigerantes, sucos em caixa, cervejas, pitú, e nescau liquido.
                           Produtos de uso diário, como papel higiênico, bombril e outros produtos."""
                            ]

# conversation_history = {}

# index_dict = 0

# index_dict_answer = 0

# dict_of_client_products = {}

pre_processed_prompt_products_list = []

# nlp = spacy.load("pt_core_news_lg")

pattern = r"\[(.*?)\]"
# pattern_for_syn = r"\{.*?\}"
pattern_for_syn = r"\{(.+?)\}"

contains_a_product = True
products_list_after_user_input = []

# model_text_generator = models.get("text-davinci-003")

## I HAVE THE DF IN MY DATABASE
# cur = conn.cursor()

# products_list = []


# cur.execute("""
#             SELECT * FROM products 
#             """)

# rows = cur.fetchall()

# 

# # 1	"#REFRIG. GUARANA ANTARCT PET CACULINHA"	"Seção 3"	1.2	"PC"	"22021000"	0.89	0.89	1.2	true
# # 2	"REFRIG. GUARANA ANTARCT PET"	"Seção 3"	6	"PC"	"22021000"	4.89	4.89	6	true

# for row in rows:
#     products = {
#         "Nome": row[1],
#         "Secao": row[2],
#         "PrecoVenda": float(row[3]),
#         "UnidadeMedida": row[4],
#         "Disponivel": row[9],
#         "Brand": row[10],
#         "Quantity": row[11],
#     }
#     products_list.append(products)


# products_df = pd.DataFrame(products_list)

# # print(products_df)


# df = pd.DataFrame(products_df)
# # df.columns = ['Nome', 'Secao', 'Disponivel']
# df.columns = ['Nome', 'Secao', 'PrecoVenda', 'UnidadeMedida', 'Disponivel','Brand', 'Quantity']

# print(df)

# embed_model = 'models/embedding-001'

#    content=f'O produto {title} é encontrado na {secao} com o preço de {preco_venda} reais possuindo a unidade de medida {unidade_medida}',  # Include the type of the product

embed_model = 'models/text-embedding-004'
def embed_fn(title, secao="", preco_venda=0, unidade_medida=""):
    
    section = find_section(title, conn)
    
    if section == None:
    
        return genai.embed_content(model=embed_model,
                                content=f'{title}',  # I understood that this have to be the most alike the userp prompt
                                title=title,
                                task_type="RETRIEVAL_DOCUMENT")["embedding"]  
    
    return genai.embed_content(model=embed_model,
                               content=f'{title} e seção {section}',  # I understood that this have to be the most alike the userp prompt
                               title=title,
                               task_type="RETRIEVAL_DOCUMENT")["embedding"]
    
    # return genai.embed_content(model=embed_model,
    #                            content=f'O produto {title}',  # I understood that this have to be the most alike the userp prompt
    #                            title=title,
    #                            task_type="RETRIEVAL_DOCUMENT")["embedding"]

# # print(embed_fn("Arroz"))


# # df["Embeddings"] = df.apply(lambda row: embed_fn(row["Nome"], row["Secao"], row["PrecoVenda"], row['UnidadeMedida']), axis=1)
# df["Embeddings"] = df.apply(lambda row: embed_fn(row["Nome"]), axis=1)

# print(df)

# # Transformar DataFrame em uma lista de tuplas
# data_tuples = [tuple(x) for x in df.to_numpy()]

# # print("Data tuples: ", data_tuples)

# sql = """INSERT INTO products_with_embeddings(
#     name,
#     secao_produto,
#     preco_venda,
#     unidade_medida,
#     disponivel,
#     brand,
#     quantity,
#     embeddings
# ) 
# values(%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *"""

# try:
#     with conn.cursor() as cur:
#         for record in data_tuples:
#             cur.execute(cur.mogrify(sql, record))

#     
# except (Exception, psycopg2.DatabaseError) as error:
#     print(error)
# finally:
#     conn.close()

## NOW I HAVE THE DF IN MY DATABASE

sql = "SELECT * FROM products_with_embeddings"

cur = conn.cursor()

cur.execute(sql)

rows = cur.fetchall()

products_list = []

for row in rows:
    products = {
        "Nome": row[1],
        "Secao": row[2],
        "PrecoVenda": float(row[3]),
        "UnidadeMedida": row[4],
        "Disponivel": row[5],
        "Embeddings": row[6],
        "Synonymous": row[7],
    }
    products_list.append(products)

products_df = pd.DataFrame(products_list)

# print(products_df)

# List of available products (I can keep that in the DB as well)
products_available = [p["Nome"] for p in products_df.to_dict(orient='records') if p['Disponivel'] > 0]

# search_term = "REF"

# matching_products = [name for name in products_available if search_term in name]

# # print("Products available: ", products_available)
# print("Matching products: ", matching_products)

def for_dividing_prompt_limit():
    products_list_2 = []
    
    for p in products_df.to_dict(orient='records'):
        # prompt += p['Nome']
        products_list_2.append(p['Nome'])
        
    max_prompt_size = 425
    
    for i in range(0, max_prompt_size):
        prompt+= products_list_2[i]

def pre_process_prompt_grammatical(prompt):
    
    fix_prompt = pre_process_prompt_grammatical_fix_prompt(prompt, business_name)
    
    time.sleep(3)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)

    # Problema agora é identificar mais de um produto :)

    return pre_processed_prompt.text

def fetch_products_alike(product_name):
    with conn.cursor() as cursor:
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        
        query = """
            SELECT name, secao_produto, preco_venda FROM products_with_embeddings
            WHERE SIMILARITY(name, %s) > 0.1;
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
    
    # fix_prompt_v1 = pre_process_prompt_grammatical(prompt)
    # print(f"FIXED 1 {fix_prompt_v1}")
    fix_prompt_v1 = prompt
    print(f"PROMPT 1 {fix_prompt_v1}")
    
    products_alike = fetch_products_alike(fix_prompt_v1)

    # print("PRODUCTS ALIKE: ", products_alike)
    
    input_embed = embed_fn(fix_prompt_v1)
    
    print("INPUT EMBED: ", input_embed)
    
    products_find_in_section = find_section(fix_prompt_v1, conn)
    products_find_with_price = find_price(fix_prompt_v1, conn)
    products_find_using_prompt = find_similar_products_using_input_embeddings(input_embed, conn, np, cosine_similarity)
    products_find_using_brand = find_brand(fix_prompt_v1, conn)
    products_find_using_metric_or_quantity = find_quantity(fix_prompt_v1, conn)
    
    fix_prompt_v2 = fix_prompt_v1
    
    if (products_find_in_section is not None 
    and products_find_with_price is not None 
    and products_find_using_brand is not None 
    and products_find_using_metric_or_quantity is not None):
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
        # Match section and price
        fix_prompt_v2 = products_match_section_and_price(
            products_find_in_section, 
            products_find_with_price, 
            products_find_using_prompt, 
            products_alike, 
            fix_prompt_v1
        )
    elif (products_find_in_section is not None and products_find_using_brand is not None):
        products_match_section_and_brand(products_find_in_section, products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1)

    elif (products_find_in_section is not None and products_find_using_metric_or_quantity is not None):
        products_match_section_and_metric_or_quantity(products_find_in_section, products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1)


    elif (products_find_with_price is not None 
        and products_find_using_brand is not None 
        and products_find_using_metric_or_quantity is not None):
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


def create_semantic_dictionary(user_input):
    """
    Creaates a semantic dictionary based on embeddings.

    Args:
        df (DataFrame): DataFrame with the products

    Returns:
        A dictionary where the keys are the products and the values are similar products lists
    """
    semantic_dictonary = {}
    # "REFRIG. GUARANA ANTARCT PET"	"Seção 3"	6	"PC"	"22021000"	4.89	4.89	6	true

    embeddings_consulta = np.array(embed_fn(user_input)).reshape(1, -1) # Reshape to 2D array for cosine similarity

    # print(f"Embed of products: {embeddings_consulta}")

    for i, row in products_df.iterrows():
        # print(f"ROW EMBED{row['Embeddings']}")
        product_embedding = np.array(row['Embeddings']).reshape(1, -1) # Ensure the product embedding is in the right shape
        
        similarity = cosine_similarity(embeddings_consulta, product_embedding)[0][0]
        
        semantic_dictonary[row["Nome"]] = similarity

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
def find_similarity_products(prompt, embeddings_products=None):
    # in teh answer I have to distinguish between the answers with products and the answerrs without products (Olá, quero saber onde fica a seção 2)

    # avatar 
    # prompt_for_detail = "Gerar uma descrição detalhada do produto [nome do produto] para um chatbot de e-commerce"
 
    # fix_response = model_2.generate_content(pre_process_prompt(prompt))

    global pre_processed_prompt_products_list
    global pattern
    global contains_a_product


    # if not process_prompt(prompt):
        

    fix_response = pre_process_prompt(prompt)

    prompt = fix_response
    
    print("CHATBOT RETURN: ", prompt)
    
    new_prompt = prompt.lower()
    new_prompt_without_spaces = new_prompt.strip()
    
    products_df['Nome'] = products_df['Nome'].str.lower()
    products_df['Nome'] = products_df['Nome'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii'))
    
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
        products_df['Cleaned_Name'] = products_df['Nome'].apply(preprocess_text)

        # Using TF-IDF to text comparation
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([new_prompt_clean] + products_df['Cleaned_Name'].tolist())  

        similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

        # Getting the products index with similarity above the threshold

        # threshold = 0.3
        threshold = 0.5
        similar_index = np.where(similarities > threshold)[0]
        sorted_index = similar_index[np.argsort(-similarities[similar_index])]
        # index = np.argsort(similarities)[0][::-1]

        similar_products = sorted_index.tolist()

        # return index[0]
        # return index[:5].tolist()
        # return index[:quantity_products].tolist()
        # print(f"INDEX LIST {index[:5].tolist()}")
        return similar_products

    contains_a_product = False
    return prompt


def generate_and_search(prompt):
   
    similarities = find_similarity_products(prompt, np.stack(products_df["Embeddings"]))

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
                
                Informações sobre o {business_name}:\n
                {info_about_the_business}
                
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
            
            cur.execute(query_add, (response, option[0]))
               
    
# user_input = "Onde fica o Refrigerante e Detergente e suco"

# user_input_processed = pre_process_prompt(user_input)

# print(user_input_processed)

# sematic_dict = create_semantic_dictionary(user_input_processed, products_df)

# # similarities = encontrar_produtos_similares(sematic_dict, 0.5)

# similarities = generate_and_search(user_input_processed)

# print(f"Produtos similares a '{similarities['Nome']}")

# def user_interaction():

    # global dict_of_client_products
    # global products_list_after_user_input
    
    # user_prompt = str(input("Olá, em que posso lhe ajudar hoje? "))

    # # falar

    # find_product = generate_and_search(user_prompt)

    # # print(process_prompt(user_prompt))

    # # print(f"TYPE: {type(find_product)}")

    # # print(find_product.iloc[0])

    # # Verificar se é um produto
    # is_product = isinstance(find_product, pd.DataFrame)

    # if is_product:
    #     count_row = find_product.shape[0]
    #     print(count_row)
    #     # if count_row > 1:
    #     if contains_a_product:
    #         # for index, row in find_product.iteritems():
    #         for index, row in find_product.iterrows():
    #             # print(f"ID: {index}, Nome: {row['Nome']}, Secao: {row['Secao']}, Disponivel: {row['Disponivel']}")
    #             dict_of_client_products = {
    #                 "Name": row['Nome'],
    #                 "Section": row['Secao'],
    #                 "Price": row['PrecoVenda'],
    #                 "Available": row['Disponivel'],
    #             }            
    #             products_list_after_user_input.append(dict_of_client_products)
                


    #         detailed_prompt = detailed_prompt = "Gerar uma descrição detalhada para um chatbot de e-commerce dos produtos que se encontram nessa lista: "
            
    #         for product in products_list_after_user_input: 
    #             detailed_prompt += f"Nome do produto: {product['Name']}\n"
    #             detailed_prompt += f"Preço do produto: {product['Price']}\n"
    #             detailed_prompt += f"Produto está disponível (False=Não, True=Sim): {product['Available']}\n\n"
            
    #         # detailed_prompt += ""
            
    #     # print(f"DICT OF PRODUCT {dict_of_client_products}")
    #     # else:
            
    #     #     ## I'M REWRINTING THE PRODUCT ON THE Name_1 space...
    #     #     dict_of_client_products["Name_1"] = find_product.iloc[0]['Nome']
    #     #     dict_of_client_products["Section_1"] = find_product.iloc[0]['Secao']
    #     #     dict_of_client_products["Available_1"] = find_product.iloc[0]['Disponivel']
            
    #     #     detailed_prompt = f"""
    #     #     Gerar uma descrição detalhada do produto {find_product['Nome']} para um chatbot de e-commerce
    #     #     """
    #     print(f"LIST OF PRODUCTS {dict_of_client_products}")
    #     generated_detailed_prompt = model_2.generate_content(detailed_prompt)

    #     # print('DETAILED PROMPT: ' + generated_detailed_prompt.text)
    

    #     # if find_product['Disponivel'].item():
    #     if find_product['Disponivel'].any():
    #         answer_prompt = f"""Considerando que o cliente do nosso
    #         mercado {supermarket_name} enviou a mensagem:

    #         {pre_process_prompt_grammatical(user_prompt)} 

    #         e os possíveis produtos que ele deseja são esses 
            
    #         """
    #         # for product in pre_processed_prompt_products_list: 
    #         for product in products_list_after_user_input: 
    #             answer_prompt += f"Nome do produto: {product['Name']}\n"
    #             answer_prompt += f"Preço do produto: {product['Price']}\n"
    #             answer_prompt += f"Seção do produto: {product['Section']}\n"
    #             answer_prompt += f"Produto está disponível (False=Não, True=Sim): {product['Available']}\n\n"

    #         # {pre_processed_prompt_products_list}.
    #         answer_prompt += f"""
    #         Considere também essa descrição detalhada dos produtos: 

    #         {generated_detailed_prompt.text} 

    #         utilize-la como um guia para melhorar a resposta para o cliente
    #         e responda com uma frase simples e útil levando em conta todos os produtos e informações disponibilizadas. Não acrescente mais nenhum texto ou caracter desnecessário 
    #         apenas a mensagem informando envolvendo algum detalhe do produto usando a descrição acima como guia e a seção que o
    #         mesmo se encontra.
            
    #         E por fim caso a mensagem do usuário: 
            
    #         {pre_process_prompt_grammatical(user_prompt)} 
            
    #         tenha alguma palavra que pareça com lista, por exemplo,
    #         'list', liste, listo, listw, gere uma possível lista de compras de acordo 
    #         com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
    #         a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
    #         não gere nenhuma lista apenas a mensagem principal.

    #         """
            
    #     else:
    #         answer_prompt = f"""Considerando que o cliente do nosso 
    #         mercado {supermarket_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
    #         que o produto que ele deseja está atualmente indisponível, 
    #         responda com uma frase simples e útil ao cliente.
    #         Considere esse texto {generated_detailed_prompt.text} como exemplo e melhore a sua resposta.
    #         E por fim caso a mensagem do usuário: 
            
    #         {pre_process_prompt_grammatical(user_prompt)} 
            
    #         tenha alguma palavra que pareça com lista, por exemplo,
    #         'list', liste, listo, listw, gere uma possível lista de compras de acordo 
    #         com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
    #         a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
    #         não gere nenhuma lista apenas a mensagem principal,
    #         leve em consideração o produto que ele queria que estava indisponível e gere
    #         uma lista que substitua o produto da melhor forma possível explicando o porque da lista"""

    # else:
    #     answer_prompt = f"""Considerando que o cliente do nosso 
    #         mercado {supermarket_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
    #         responda com uma frase simples e útil ao cliente. 
    #         E por fim gere alguma evento ou ação para o cliente fazer
    #         no mercado se vier a o caso, por exemplo, "Queria saber que horas o
    #         mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
    #         """
        
    # response = model_2.generate_content(answer_prompt)

    # # global index_dict_answer

    # # index_dict_answer = index_dict_answer + 1
    # # name = f"Answer_{index_dict_answer}"

    # # conversation_history[name] = response.text

    # sql = """INSERT INTO conversation_history (
    #         user_input,
    #         chatbot_response)
    #         VALUES (%s, %s) RETURNING *"""
        
    # try:
    #     with conn.cursor() as cur:
    #         cur.execute(sql, (pre_process_prompt_grammatical(user_prompt), response.text))
    #         
            
    # except (Exception, psycopg2.DatabaseError) as error:
    #     print(error)
            
    # # finally:
    # #     conn.close()
            
    # print(response.text)

    # while True:
    #     # user_prompt = str(input("Posso ajudar em algo mais? "))

    #     if process_prompt(pre_process_prompt_grammatical(user_prompt)):
    #         conn.close()
    #         break

    #     # find_product = generate_and_search(user_prompt)
    
    #     is_product = isinstance(find_product, pd.DataFrame)

    
    #     if is_product:
    #         count_row = find_product.shape[0]
    #         print(count_row)
    #         # if count_row > 1:
    #         if contains_a_product:
    #             # for index, row in find_product.iteritems():
    #             for index, row in find_product.iterrows():
    #                 # print(f"ID: {index}, Nome: {row['Nome']}, Secao: {row['Secao']}, Disponivel: {row['Disponivel']}")
    #                 dict_of_client_products = {
    #                     "Name": row['Nome'],
    #                     "Section": row['Secao'],
    #                     "Price": row['PrecoVenda'],
    #                     "Available": row['Disponivel'],
    #                 }            
    #                 products_list_after_user_input.append(dict_of_client_products)
                    


    #             detailed_prompt = detailed_prompt = "Gerar uma descrição detalhada para um chatbot de e-commerce dos produtos que se encontram nessa lista: "
                    
    #             for product in products_list_after_user_input: 
    #                 detailed_prompt += f"Nome do produto: {product['Name']}\n"
    #                 detailed_prompt += f"Preço do produto: {product['Price']}\n"
    #                 detailed_prompt += f"Produto está disponível (False=Não, True=Sim): {product['Available']}\n\n"
             

    #         generated_detailed_prompt = model_2.generate_content(detailed_prompt)

    #         # if find_product['Disponivel'].item():
    #         if find_product['Disponivel'].any():
    #             answer_prompt = f"""Considerando que o cliente do nosso
    #             mercado {supermarket_name} enviou a mensagem:

    #             {pre_process_prompt_grammatical(user_prompt)} 

    #             e os possíveis produtos que ele deseja são esses 
                
    #             """
    #             # for product in pre_processed_prompt_products_list: 
    #             for product in products_list_after_user_input: 
    #                 answer_prompt += f"Nome do produto: {product['Name']}\n"
    #                 answer_prompt += f"Preço do produto: {product['Price']}\n"
    #                 answer_prompt += f"Seção do produto: {product['Section']}\n"
    #                 answer_prompt += f"Produto está disponível (False=Não, True=Sim): {product['Available']}\n\n"

    #             # {pre_processed_prompt_products_list}.
    #             answer_prompt += f"""
    #             Considere também essa descrição detalhada dos produtos: 

    #             {generated_detailed_prompt.text} 

    #             utilize-la como um guia para melhorar a resposta para o cliente
    #             e responda com uma frase simples e útil levando em conta todos os produtos e informações disponibilizadas. Não acrescente mais nenhum texto ou caracter desnecessário 
    #             apenas a mensagem informando envolvendo algum detalhe do produto usando a descrição acima como guia e a seção que o
    #             mesmo se encontra.
    #             E por fim caso a mensagem do usuário: 
                
    #             {pre_process_prompt_grammatical(user_prompt)} 
                
    #             tenha alguma palavra que pareça com lista, por exemplo,
    #             'list', liste, listo, listw, gere uma possível lista de compras de acordo 
    #             com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
    #             a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
    #             não gere nenhuma lista apenas a mensagem principal.

    #             """
                
    #         else:
    #             answer_prompt = f"""Considerando que o cliente do nosso 
    #             mercado {supermarket_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
    #             que o produto que ele deseja está atualmente indisponível, 
    #             responda com uma frase simples e útil ao cliente.
    #             Considere esse texto {generated_detailed_prompt.text} como exemplo e melhore a sua resposta.
    #             E por fim caso a mensagem do usuário: 
                
    #             {pre_process_prompt_grammatical(user_prompt)} 
                
    #             tenha alguma palavra que pareça com lista, por exemplo,
    #             'list', liste, listo, listw, gere uma possível lista de compras de acordo 
    #             com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
    #             a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
    #             não gere nenhuma lista apenas a mensagem principal,
    #             leve em consideração o produto que ele queria que estava indisponível e gere
    #             uma lista que substitua o produto da melhor forma possível explicando o porque da lista"""

    #     else:
    #         answer_prompt = f"""Considerando que o cliente do nosso 
    #             mercado {supermarket_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
    #             responda com uma frase simples e útil ao cliente. 
    #             E por fim gere alguma evento ou ação para o cliente fazer
    #             no mercado se vier a o caso, por exemplo, "Queria saber que horas o
    #             mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
    #             """
        

    #     response = model_2.generate_content(answer_prompt)

    #     # print(f"ANSWER PROMPT {answer_prompt}")

    #     # print(response.text)

    #     # index_dict_answer = index_dict_answer + 1
    #     # name = f"Answer_{index_dict_answer}"

    #     # conversation_history[name] = response.text
    #     # print(f"\nCONVERSATION HISTORY - {conversation_history}\n")
        
    #     sql = """INSERT INTO conversation_history (
    #         user_input,
    #         chatbot_response)
    #         VALUES (%s, %s) RETURNING *"""
        
    #     try:
    #         with conn.cursor() as cur:
    #             cur.execute(sql, (pre_process_prompt_grammatical(user_prompt), response.text))
    #         
            
    #     except (Exception, psycopg2.DatabaseError) as error:
    #         print(error)
            
    #     # finally:
    #     #     conn.close()           
            

# user_interaction()

# um algoritmo que descreve a inclusão de um novo topico



def chatbot_response(user_prompt):
    
    global dict_of_client_products
    global products_list_after_user_input

    fix_prompt_v1 = pre_process_prompt_grammatical(user_prompt)

    find_product = generate_and_search(fix_prompt_v1)

    is_product = isinstance(find_product, pd.DataFrame)
    
    print("USER PROMPT: ", user_prompt)
    print("Fixed prompt v1: ", fix_prompt_v1)

    if is_product:
        count_row = find_product.shape[0]
        print(count_row)
        # if count_row > 1:
        if contains_a_product:
            # for index, row in find_product.iteritems():
            for index, row in find_product.iterrows():
                # print(f"ID: {index}, Nome: {row['Nome']}, Secao: {row['Secao']}, Disponivel: {row['Disponivel']}")
                dict_of_client_products = {
                    "Name": row['Nome'],
                    "Section": row['Secao'],
                    "Price": row['PrecoVenda'],
                    "Available": row['Disponivel'],
                }            
                products_list_after_user_input.append(dict_of_client_products)                
        
            detailed_prompt = detailed_prompt = "Gerar uma descrição detalhada para um chatbot de e-commerce dos produtos que se encontram nessa lista: "
                
            name_of_product = ""
            matching_products = []
            for product in products_list_after_user_input: 
                name_of_product = product['Name']
                detailed_prompt += f"Nome do produto: {product['Name']}\n"
                detailed_prompt += f"Preço do produto: {product['Price']}\n"
                detailed_prompt += f"Produto está disponível com {product['Available']} em estoque\n\n"
            
            print("Name of product: ",name_of_product)
            if (len(name_of_product) > 0):
                first_letter_of_product = name_of_product[:4]
                
                print("Four first letters: ", first_letter_of_product)

                matching_products = [name for name in products_available if first_letter_of_product.upper() in name]
            else:
                matching_products = products_available[:20]
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

        # if find_product['Disponivel'].item():
        if find_product['Disponivel'].any():
            answer_prompt = f"""Considerando que o cliente do nosso
            mercado {business_name} enviou a mensagem:

            {pre_process_prompt_grammatical(user_prompt)} 

            e os possíveis produtos que ele deseja são esses 
            
            """
            # for product in pre_processed_prompt_products_list: 
            for product in products_list_after_user_input: 
                answer_prompt += f"Nome do produto: {product['Name']}\n"
                answer_prompt += f"Preço do produto: {product['Price']}\n"
                answer_prompt += f"Seção do produto: {product['Section']}\n"
                answer_prompt += f"Produto está disponível com {product['Available']} em estoque\n\n"

            # {pre_processed_prompt_products_list}.
            answer_prompt += f"""
            Considere também essa descrição detalhada dos produtos: 

            {generated_detailed_prompt.text} 

            utilize-la como um guia para melhorar a resposta para o cliente
            e responda com uma frase simples e útil levando em conta todos os produtos e informações disponibilizadas. Não acrescente mais nenhum texto ou caracter desnecessário 
            apenas a mensagem informando envolvendo algum detalhe do produto usando a descrição acima como guia e a seção que o
            mesmo se encontra.
            
            Além dessas informações sobre o {business_name}:
            """
            
            for info in info_about_the_business:
                print(f"INFO: {info}")
                answer_prompt+=f"{info}\n"
        
            answer_prompt+=f"""
            E por fim caso a mensagem do usuário: 
            
            {pre_process_prompt_grammatical(user_prompt)} 
            
            tenha alguma palavra que pareça com lista, por exemplo,
            'list', liste, listo, listw, gere uma possível lista de compras de acordo 
            com os produtos disponíveis no mercado possivelmente parecidos com o produto solicitado pelo usário sendo esses:
            
            {matching_products} 
            
            Repetindo, caso se e somente se
            a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
            não gere nenhuma lista apenas a mensagem principal.

            """
            
        else:
            answer_prompt = f"""Considerando que o cliente do nosso 
            mercado {business_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
            que o produto que ele deseja está atualmente indisponível, 
            responda com uma frase simples e útil ao cliente.
            Considere esse texto {generated_detailed_prompt.text} como exemplo e melhore a sua resposta.
            E por fim caso a mensagem do usuário: 
            
            {pre_process_prompt_grammatical(user_prompt)} 
            
            tenha alguma palavra que pareça com lista, por exemplo,
            'list', liste, listo, listw, gere uma possível lista de compras de acordo 
            com os produtos disponíveis no mercado possivelmente parecidos com o produto solicitado pelo usário sendo esses:
            
            {matching_products} 
            
            Repetindo, caso se e somente se
            a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
            não gere nenhuma lista apenas a mensagem principal,
            leve em consideração o produto que ele queria que estava indisponível e gere
            uma lista que substitua o produto da melhor forma possível explicando o porque da lista"""

    else:
        
        products_alike = fetch_products_alike(user_prompt)
        
        if user_prompt == "Saber mais sobre os produtos":
            answer_prompt = f"""
                Considerando que o cliente do nosso
                mercado {business_name} enviou a mensagem:

                {user_prompt} 

                e os possíveis produtos temos disponíveis são esses: 
                \n
                """
                
            with conn.cursor() as cursor:
                
                sql_query = """
                SELECT name, secao_produto, preco_venda, disponivel FROM products_with_embeddings
                ORDER BY RANDOM() LIMIT 20;
                """
                
                cursor.execute(sql_query)
                result = cursor.fetchall()
                
                
            
            product_info = ""
            
            for product in result:
                for i in range(len(product)):
                    # print("RANDOM PRODUCT: ", product[i])
                    product_info += str(product[i]) + " " 
                
                answer_prompt+=product_info.strip() + "\n"
            
            print("Fixed prompt V1: ", fix_prompt_v1)                
                
            
            answer_prompt+="""
            Considerando esses produtos que contém os seguintes valores nome do produto, seção do produto, preco de venda e se o mesmo está disponivel 
            extamente nessa sequência, retorne uma mensagem com alguns produtos que temos para ver se o cliente vai querer algum deles
            retorne a melhor resposta possível.
            """
            
        elif user_prompt == "Saber mais sobre nosso mercado":
            answer_prompt = f"""
                Considerando que o cliente do nosso
                mercado {business_name} enviou a mensagem:

                {user_prompt} 

                e as informações sobre o {business_name} que temos disponíveis são essas: 
                \n
                """
                
            for info in info_about_the_business:
                answer_prompt+=f"{info}\n"
            
        else:
            
            answer_prompt = f"""Considerando que o cliente do nosso 
                mercado {business_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
                responda com uma frase simples e útil ao cliente. 
                E por fim gere alguma evento ou ação para o cliente fazer
                no mercado se vier a o caso, por exemplo, "Queria saber que horas o
                mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
                
                Leve em consideração essas informações sobre o {business_name}:
                """
            
            for info in info_about_the_business:
                answer_prompt+=f"{info}\n"
        

    section_products = find_section(fix_prompt_v1, conn)
    
    if section_products:
        answer_prompt+=f"""
        Levando em consideração também que esses produtos:
        
        {section_products}
        
        São os produtos disponíveis na seção que o cliente comentou retorne uma resposta conveniente com alguns exemplos usando esses produtos para o mesmo. Em caso do dado acima ter alguma mensagem parecida com "This section isn't available in our establishment" retorne informando ao cliente sobre esse problema relacionado a seção.
                """
    
    answer_prompt+="""
            Considerando essas informações retorne uma mensagem com útil para o cliente e alguma possível ação ou pergunta que o cliente pode fazer agora.
            retorne a melhor resposta possível.
            """


    response = chat.send_message(answer_prompt)

    print(f"ANSWER PROMPT {answer_prompt}")
    
    
    print("-"*80)
    print("RESPONSE: ", response.text)

    # return f"'{response.text}'"
    return response.text


# print(chatbot_response("VITAMASSA"))

def generate_synonymous(product=None, product_list=None):
    products_dict = {}
    product_synonymous_list = []
    
    if product_list:
        print("product list: ", product_list)
    
        print("product list name: ", product_list.name)
        original_product = product_list.name
        
        # semantic_dictorinary = create_semantic_dictionary(product)
        products = generate_and_search(original_product)
        
        
        print("PRODUCTS SIZE: ", products.size)
        
        if products.size > 1:
            for index, row in products.iterrows():

                products_dict = {
                    "Name": row['Nome'],
                    "Section": row['Secao'],
                    "Price": row['PrecoVenda'],
                    "Available": row['Disponivel'],
                }            
                product_synonymous_list.append(products_dict)
                # print("GENERATE SYNONYMOUS PRODUCT", products_dict["Name"])
    
    else:
        print("product: ", product)
        
        print("product name: ", product)
        original_product = product
        
        # semantic_dictorinary = create_semantic_dictionary(product)
        products = generate_and_search(product)
        
        print("PRODUCTS SIZE: ", products.size)
    
    if products.size > 1:
        for index, row in products.iterrows():

            products_dict = {
                "Name": row['Nome'],
                "Section": row['Secao'],
                "Price": row['PrecoVenda'],
                "Available": row['Disponivel'],
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
        new_product = pre_process_prompt_grammatical(original_product)
        
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
        
        for product in product_synonymous_list:
            prompt += str(product["Name"]) + "\n"
        
        prompt += """\n\nOutput example (adjusted for dynamic products):
        
        {"bisc. rosq. chocolate": ["biscuit chocolate ring", "cookie chocolate ring", "ring-shaped chocolate biscuit"], "bisc. rosq. leite": ["milk biscuit ring", "milk cookie ring", "ring-shaped milk biscuit"]}
        
        Important Notes:

        1. Do not include duplicate synonyms across or within products.
        
        2. Do not return any extra characters, comments, or text (e.g., no json, python, or similar formatting).
        
        3. The product list may change dynamically. Always adapt to the provided product names.
        
        4. ONLY return the synonymous don't return any other text, for example ```python ```json or anything like that, I want the response to start with a { and end with a }, the format that I've just told you about.
        """ 

    
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
                        """, (sanitized_synonymous_list, original_product))
            
                
        return sanitized_synonymous_list
    
    except Exception as e:
        print("Error generating synonyms: ", e)
        return None

# print(generate_synonymous("REFRIG. GUARANA  PET"))
