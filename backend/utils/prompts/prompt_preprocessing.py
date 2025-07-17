from utils.prompts_utils import * 

import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

def handle_all_conditions(ctx):
    return products_match_all_conditions(ctx["section"], ctx["price"],
        ctx["brand"],
        ctx["quantity"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],)
    
def handle_section_price_and_brand(ctx):
    return products_match_section_price_and_brand(
        ctx["section"],
        ctx["price"],
        ctx["brand"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_price_section_and_quantity(ctx):
    return products_match_price_brand_and_metric(
        ctx["price"],
        ctx["section"],
        ctx["quantity"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_section_and_price(ctx):
    return products_match_section_and_price(
        ctx["section"],
        ctx["price"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_section_and_quantity(ctx):
    return products_match_section_and_metric_or_quantity(
        ctx["section"],
        ctx["quantity"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_section_and_brand(ctx):
    return products_match_section_and_brand(
        ctx["section"],
        ctx["brand"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_price_and_brand(ctx):
    return products_match_price_and_brand(
        ctx["price"],
        ctx["brand"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_price_and_quantity_or_metric(ctx):
    return products_match_price_and_quantity_or_metric(
        ctx["price"],
        ctx["quantity"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )

def handle_only_section(ctx):
    return products_find_in_section_fix_prompt(
        ctx["section"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )

def handle_only_brand(ctx):
    return products_find_with_brand_fix_prompt(
        ctx["brand"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )

def handle_only_price(ctx):
    return products_find_with_price_fix_prompt(
        ctx["price"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )

def handle_only_quantity(ctx):
    return products_find_with_metric_or_quantity_fix_prompt(
        ctx["quantity"],
        ctx["sim_embeddings"],
        ctx["alike"],
        ctx["raw"],
    )
    
def handle_general(ctx):
    return products_find_general_fix_prompt(
        ctx["alike"],
        ctx["sim_embeddings"],
        ctx["raw"],
    )

CONDITIONS = [
    # all four: section, price, brand, quantity
    (
        lambda ctx: ctx["section"] is not None
                 and ctx["price"]   is not None
                 and ctx["brand"]   is not None
                 and ctx["quantity"] is not None,
        handle_all_conditions
    ),

    # section + price + brand
    (
        lambda ctx: ctx["section"] is not None
                 and ctx["price"]   is not None
                 and ctx["brand"]   is not None,
        handle_section_price_and_brand
    ),

    # section + price + quantity
    (
        lambda ctx: ctx["section"] is not None
                 and ctx["price"]   is not None
                 and ctx["quantity"]   is not None,
        handle_price_section_and_quantity
    ),

    # section + price only
    (
        lambda ctx: ctx["section"] is not None
                 and ctx["price"]   is not None,
        handle_section_and_price
    ),

    # section + quantity only
    (
        lambda ctx: ctx["section"] is not None
                 and ctx["quantity"]   is not None,
        handle_section_and_price
    ),

    # section + brand only
    (
        lambda ctx: ctx["section"] is not None
                 and ctx["brand"]   is not None,
        handle_section_and_brand
    ),

    # price + brand only
    (
        lambda ctx: ctx["price"] is not None
                 and ctx["brand"]   is not None,
        handle_price_and_brand
    ),

    # price + quantity only
    (
        lambda ctx: ctx["price"] is not None
                 and ctx["quantity"]   is not None,
        handle_price_and_quantity_or_metric
    ),

    # only section
    (
        lambda ctx: ctx["section"] is not None,
        handle_only_section
    ),

    # only brand
    (
        lambda ctx: ctx["brand"] is not None,
        handle_only_brand
    ),

    # only price
    (
        lambda ctx: ctx["price"] is not None,
        handle_only_price
    ),

    # only quantity
    (
        lambda ctx: ctx["quantity"] is not None,
        handle_only_quantity
    ),
    # Fallback (always matches last)
    (
        lambda ctx: True,
        handle_general  # your products_find_general_fix_prompt wrapper
    ),
]


def pre_process_prompt(prompt, find_section, find_price, find_brand, find_quantity, find_using_synonyms, find_similar_products_using_input_embeddings, embed_fn, fetch_products_alike, generate_content_model, conn, business=None):
    ctx = {
        "raw": prompt,
        "section": find_section(prompt, business_id=business),
        "price": find_price(prompt, business_id=business),
        "brand": find_brand(prompt, business_id=business),
        "quantity": find_quantity(prompt, business_id=business),
        "synonyms": find_using_synonyms(prompt, conn),
        "sim_embeddings": find_similar_products_using_input_embeddings(embed_fn(find_section, genai, business, prompt), business_id=business),
        "alike": fetch_products_alike(prompt),
    }
    
    for predicate, handler in CONDITIONS:
        if predicate(ctx):
            fix_prompt = handler(ctx)
            print("NEW fix prompt: ", fix_prompt)
            break
        
    time.sleep(3)
    
    return generate_content_model.generate_content(fix_prompt).text

def pre_process_prompt_brand(prompt, brands_find, business_name, generate_content_model):
    fix_prompt = pre_process_prompt_grammatical_brand_fix_prompt(prompt, business_name, brands_find)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)
    
    # time.sleep(3)
    return pre_processed_prompt.text

def pre_process_prompt_section(prompt, find_section, find_similar_products_using_input_embeddings, conn, embed_fn, fetch_products_alike, generate_content_model, business):
    products_find_in_section = find_section(prompt, business_id=business)
    input_embed = embed_fn(find_section, genai, business, prompt)
    products_find_using_prompt = find_similar_products_using_input_embeddings(input_embed=input_embed, business_id=business)
    products_alike = fetch_products_alike(prompt)
    
    fix_prompt = products_find_in_section_fix_prompt(products_find_in_section, products_find_using_prompt, products_alike, prompt)

    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)
    
    return pre_processed_prompt.text

def pre_process_prompt_grammatical(prompt, business_name, generate_content_model):
    
    fix_prompt = pre_process_prompt_grammatical_fix_prompt(prompt, business_name)
    
    time.sleep(3)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)

    # Problema agora é identificar mais de um produto :)

    return pre_processed_prompt.text


def create_menu_options_prompts(fetch_products_alike, conn, generate_content_model):
    """
    Function to create the menu options based on the user registration (NOT USED FOR NOW)
    """
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
               
