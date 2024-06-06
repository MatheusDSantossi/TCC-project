import google.generativeai as genai
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

import unicodedata

# import sys
# import os

# # Addings the folder path 'config'
# config_path = "D:\\matheus_lucy_mkd\\Documents\\exercicios_faculdade\\TCC_stuff\\new_idea_ecommerce\\config.py"

# # Adiciona o diretório pai do arquivo config.py ao sys.path
# sys.path.append(os.path.dirname(config_path))

# import config

# Configurando a API do Gemini
# GOOGLE_API_KEY = config.SECRET_KEY
GOOGLE_API_KEY = 'AIzaSyBskS4hcTWUP0L0QROEbuPEKdvp4KVlWkQ'
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.4,
    "candidate_count": 1,
}

model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)


# nltk.data.clear_cache()

supermarket_name = "Supermercado Santos"

# Products DataFrame
products = [
    {"Nome": "Café Ouro Verde", "Seção": "Seção 1", "Disponivel": True},
    {"Nome": "Leite Camponesa", "Seção": "Seção 2", "Disponivel": True},
    {"Nome": "Arroz Caçarola", "Seção": "Seção 2", "Disponivel": False},
    {"Nome": "Feijão Carioca", "Seção": "Seção 2", "Disponivel": True},
    {"Nome": "Macarrão Barilla", "Seção": "Seção 2", "Disponivel": True},
    {"Nome": "Suco de Laranja Natural", "Seção": "Seção 3", "Disponivel": True},
    {"Nome": "Refrigerante Coca-Cola", "Seção": "Seção 3", "Disponivel": True},
    {"Nome": "Água Mineral", "Seção": "Seção 3", "Disponivel": True},
    {"Nome": "Ovos Brancos", "Seção": "Seção 4", "Disponivel": True},
    {"Nome": "Frango Congelado", "Seção": "Seção 4", "Disponivel": True},
    {"Nome": "Banana Nanica", "Seção": "Seção 5", "Disponivel": True},
    {"Nome": "Maçã Gala", "Seção": "Seção 5", "Disponivel": True},
    {"Nome": "Tomate", "Seção": "Seção 5", "Disponivel": True},
    {"Nome": "Cebola", "Seção": "Seção 5", "Disponivel": True},
    {"Nome": "Sabão em Pó Omo", "Seção": "Seção 6", "Disponivel": True},
    {"Nome": "Detergente Limpol", "Seção": "Seção 6", "Disponivel": True},
    {"Nome": "Papel Higiênico", "Seção": "Seção 7", "Disponivel": True},
    {"Nome": "Shampoo Head & Shoulders", "Seção": "Seção 8", "Disponivel": True},
    {"Nome": "Pasta de Dente Colgate", "Seção": "Seção 8", "Disponivel": True},
    {"Nome": "Biscoito Negresco", "Seção": "Seção 9", "Disponivel": True},
]


products_df = pd.DataFrame(products)

# List of available products
products_available = [p["Nome"] for p in products_df.to_dict(orient='records') if p['Disponivel']]

df = pd.DataFrame(products_df)
df.columns = ['Nome', 'Secao', 'Disponivel']

# print(df)

model = 'models/embedding-001'

def embed_fn(title, text):
    return genai.embed_content(model=model,
                               content=f'O produto {title} é encontrado na {text}',  # Include the type of the product
                               title=title,
                               task_type="RETRIEVAL_DOCUMENT")["embedding"]

df["Embeddings"] = df.apply(lambda row: embed_fn(row["Nome"], row["Secao"]), axis=1)

# print(df)

def find_similarity_products(prompt, embeddings_products):
    # in teh answer I have to distinguish between the answers with products and the answerrs without products (Olá, quero saber onde fica a seção 2)

    # avatar 

    fix_prompt = f"""
        Corrija os erros gramaticais e ortográficos no seguinte texto para que ele seja compreensível para um leitor 
        humano e normalize os nomes dos produtos para a forma padrão, conforme encontrado na lista de produtos abaixo:

        Lista de produtos:

        Café Ouro Verde
        Leite Camponesa
        Arroz Caçarola
        Feijão Carioca
        Macarrão Barilla
        Suco de Laranja Natural
        Refrigerante Coca-Cola
        Água Mineral
        Ovos Brancos
        Frango Congelado
        Banana Nanica
        Maçã Gala
        Tomate
        Cebola
        Sabão em Pó Omo
        Detergente Limpol
        Papel Higiênico
        Shampoo Head & Shoulders
        Pasta de Dente Colgate
        Biscoito Negresco

        Texto:

        "{prompt}"

        Mantenha o significado original do texto e corrija apenas os erros de escrita e gramática. Por exemplo, "cafezinho" 
        deve ser corrigido para "Café Ouro Verde", "leitinho" para "Leite Camponesa", e assim por diante. Seja o mais direto 
        possível, sem acrescentar informações adicionais.

        Regras de Retorno:

        Caso o texto não contenha nenhum produto ou seja algo totalmente não relacionado aos produtos, como 
        "Olá, como você está?" ou "Queria saber que horas o mercado abre?", retorne "Texto RA".
    """
    # In case the product is not in the list, show the message "Produto não está disponível!"
    
    # Problema agora é identificar mais de um produto :)

    fix_response = model_2.generate_content(fix_prompt)

    prompt = fix_response.text
    
    print(prompt)
    
    new_prompt = prompt.lower()
    
    new_prompt = unicodedata.normalize('NFKD', prompt).encode('ascii', 'ignore').decode('ascii')
    
    products_df['Nome'] = products_df['Nome'].str.lower()
    products_df['Nome'] = products_df['Nome'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii'))
    
    if (prompt not in ['Produto NA', 'Texto RA']):

        # print("NOT IN NA")

        # products_df['Nome'] = products_df['Nome'].str.lower().str.replace('ã', 'a').str.replace('é', 'e').str.replace('õ', 'o')
        

        # Using TF-IDF to text comparation
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([new_prompt] + products_df['Nome'].tolist())  

        similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
        index = np.argsort(similarities)[0][::-1]

        return index[0]

    else: 

        return prompt

def generate_and_search(prompt):
    # processed_prompt = process_prompt(prompt) # vamos ignorar por enquanto, porque não está puxando porque no nltk está errado

    # embedding_prompt = genai.embed_content(model=model,
    #                                        content=processed_prompt,
    #                                        task_type="RETRIEVAL_DOCUMENT")["embedding"]

    # similarities = find_similarity_products(embedding_prompt, np.stack(df["Embeddings"]))
    # similarities = find_similarity_products(processed_prompt, np.stack(df["Embeddings"]))
    
    # new_prompt = prompt.lower().replace('ã', 'a').replace('é', 'e').replace('õ', 'o')
    # similarities = find_similarity_products(prompt, np.stack(df["Embeddings"]))

    similarities = find_similarity_products(prompt, np.stack(df["Embeddings"]))

    # if similarities not in ['Produto NA', 'Texto RA']:        

    #     # return df.iloc[similarities]
    #     return df.iloc[[similarities]]

    if isinstance(similarities, np.int64):  

        return df.iloc[[similarities]]

    return similarities

# Process the prompt using NLTK 
def process_prompt(prompt):
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(prompt)
    pos_tags = pos_tag(tokens)
    
    # Extract the named entities (people, place, things)
    nouns = [lemmatizer.lemmatize(token) for token, tag in pos_tags if tag == 'NN']

    # Extract the verbs (to identify the wished action)
    verbs = [lemmatizer.lemmatize(token) for token, tag in pos_tags if tag.endswith('VB')]

    # Create a representation of the prompt based on the entities and verbs
    prompt_representation = " ".join(nouns) + " " + " ".join(verbs)
    
    return prompt_representation

# print(process_prompt('Onde posso encontrar café?'))

user_prompt = str(input("Olá, em que posso lhe ajudar hoje? "))

# falar

find_product = generate_and_search(user_prompt)

# print(process_prompt(user_prompt))

print(f"TYPE: {type(find_product)}")

# Verificar se é um produto
is_product = isinstance(find_product, pd.DataFrame)

if is_product:        

    if find_product['Disponivel'].item():
        answer_prompt = f"""Considerando que o cliente do nosso
        mercado {supermarket_name} quer saber {user_prompt} 
        que o produto que ele deseja se encontra na seção {find_product['Secao'].item()}, 
        responda com uma frase simples e útil ao cliente. 
        E por fim gere uma possível lista de compras de acordo 
        com os produtos disponíveis no mercado que são esses, {products_available}.

        """
        
    else:
        answer_prompt = f"""Considerando que o cliente do nosso 
        mercado {supermarket_name} quer saber {user_prompt} 
        que o produto que ele deseja está atualmente indisponível, 
        responda com uma frase simples e útil ao cliente. 
        E por fim gere uma possível lista de compras de acordo 
        com os produtos disponíveis no mercado que são esses, {products_available}, porém,
        leve em consideração o produto que ele queria e gere
        uma lista que substitua o produto da melhor forma possível"""

else:
    answer_prompt = f"""Considerando que o cliente do nosso 
        mercado {supermarket_name} quer saber {user_prompt} 
        responda com uma frase simples e útil ao cliente. 
        E por fim gere alguma evento ou ação para o cliente fazer
        no mercado se vier a o caso, por exemplo, "Queria saber que horas o
        mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
        """


response = model_2.generate_content(answer_prompt)

print(response.text)

while True:
    user_prompt = str(input("Posso ajudar em algo mais? "))
    

    # print(find_product)

    # # if find_product['Disponivel'].bool(): 
    # if find_product['Disponivel'].item(): 
    #     print(True)

    if(user_prompt.lower() in ['sair', 'exit','no']):
        print('Até logo!')
        break

    find_product = generate_and_search(user_prompt)
  
    is_product = isinstance(find_product, pd.DataFrame)

    if is_product:   

        if find_product['Disponivel'].item():
            answer_prompt = f"""Considerando que o cliente do nosso
            mercado {supermarket_name} quer saber {process_prompt(user_prompt)} 
            que o produto que ele deseja se encontra na seção {find_product['Secao'].item()}, 
            responda com uma frase simples e útil ao cliente. 
            E por fim gere uma possível lista de compras de acordo 
            com os produtos disponíveis no mercado que são esses, {products_available}.

            """
            
        else:
            answer_prompt = f"""Considerando que o cliente do nosso 
            mercado {supermarket_name} quer saber {process_prompt(user_prompt)} 
            que o produto que ele deseja está atualmente indisponível, 
            responda com uma frase simples e útil ao cliente. 
            E por fim gere uma possível lista de compras de acordo 
            com os produtos disponíveis no mercado que são esses, {products_available}, porém,
            leve em consideração o produto que ele queria e gere
            uma lista que substitua o produto da melhor forma possível"""

    else:
        answer_prompt = f"""Considerando que o cliente do nosso 
            mercado {supermarket_name} quer saber {user_prompt} 
            responda com uma frase simples e útil ao cliente. 
            E por fim gere alguma evento ou ação para o cliente fazer
            no mercado se vier a o caso, por exemplo, "Queria saber que horas o
            mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
            """

    response = model_2.generate_content(answer_prompt)

    print(response.text)
