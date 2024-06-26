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

from config import SECRET_KEY

from fuzzywuzzy import fuzz

# Configurando a API do Gemini
GOOGLE_API_KEY = SECRET_KEY
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.4,
    "candidate_count": 1,
}

model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)


# nltk.data.clear_cache()

supermarket_name = "Supermercado Santos"

conversation_history = {}

index_dict = 0

global index_dict_answer

index_dict_answer = 0


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
    {"Nome": "Tomate", "Seção": "Seção 5", "Disponivel": False},
    {"Nome": "Cebola", "Seção": "Seção 5", "Disponivel": True},
    {"Nome": "Sabão em Pó Omo", "Seção": "Seção 6", "Disponivel": True},
    {"Nome": "Detergente Limpol", "Seção": "Seção 6", "Disponivel": False},
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

def pre_process_prompt(prompt):
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

        Mantenha o significado original do texto e corrija apenas os erros de escrita e gramática. Por exemplo, "quero um cafezinho" 
        deve ser corrigido para "Quero um Café Ouro Verde", "Tem leitinho" para "Tem Leite Camponesa", "Quero pasta" para "Quero Pasta de Dente Colgate" e assim por diante. 
        Seja o mais direto possível, sem acrescentar informações adicionais.

        Regras de Retorno:

        Caso o texto não contenha nenhum produto ou seja algo totalmente não relacionado aos produtos, como 
        "Olá, como você está?" ou "Queria saber que horas o mercado abre?", "Como é o seu nome" retorne "Texto RA".
    """
    # In case the product is not in the list, show the message "Produto não está disponível!"
    
    pre_processed_prompt = model_2.generate_content(fix_prompt)

    # Problema agora é identificar mais de um produto :)

    return pre_processed_prompt.text

def find_similarity_products(prompt, embeddings_products):
    # in teh answer I have to distinguish between the answers with products and the answerrs without products (Olá, quero saber onde fica a seção 2)

    # avatar 
    # prompt_for_detail = "Gerar uma descrição detalhada do produto [nome do produto] para um chatbot de e-commerce"
 
    # fix_response = model_2.generate_content(pre_process_prompt(prompt))

    if not process_prompt(prompt):
        

        fix_response = pre_process_prompt(prompt)

        prompt = fix_response
        
        print(prompt)
        # conversation_history.append(prompt)
        global index_dict 
        index_dict += 1
        name = f"Question_{index_dict}"
        conversation_history[name] = prompt
        
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

    # lemmatizer = WordNetLemmatizer()

    # tokens = word_tokenize(prompt)
    # pos_tags = pos_tag(tokens)
    
    # # Extract the named entities (people, place, things)
    # nouns = [lemmatizer.lemmatize(token) for token, tag in pos_tags if tag == 'NN']

    # # Extract the verbs (to identify the wished action)
    # verbs = [lemmatizer.lemmatize(token) for token, tag in pos_tags if tag.endswith('VB')]

    # # Create a representation of the prompt based on the entities and verbs
    # prompt_representation = " ".join(nouns) + " " + " ".join(verbs)
    
    # return prompt_representation
    if fuzz.token_set_ratio(prompt.lower(), "quero sair") >= 80:
        # print(process_prompt(user_prompt.lower()))  # Processa antes de sair
        print('Até logo!')
        return True  # Sinaliza para sair do loop

    return False  # Continua a conversa

def user_interaction():
    
    user_prompt = str(input("Olá, em que posso lhe ajudar hoje? "))

    # falar

    find_product = generate_and_search(user_prompt)

    # print(process_prompt(user_prompt))

    # print(f"TYPE: {type(find_product)}")

    # Verificar se é um produto
    is_product = isinstance(find_product, pd.DataFrame)

    if is_product:       
        detailed_prompt = f"""
        Gerar uma descrição detalhada do produto {find_product['Nome']} para um chatbot de e-commerce
        """

        generated_detailed_prompt = model_2.generate_content(detailed_prompt)

        print('DETAILED PROMPT: ' + generated_detailed_prompt.text)
    

        if find_product['Disponivel'].item():
            answer_prompt = f"""Considerando que o cliente do nosso
            mercado {supermarket_name} enviou a mensagem:

            {pre_process_prompt(user_prompt)} 

            e o possível produto que ele deseja se encontra na seção {find_product['Secao'].item()}, 
            Considere essa descrição detalhada desse produto: 

            {generated_detailed_prompt.text} 

            utilize-la como um guia para melhorar a resposta para o cliente
            e responda com uma frase simples e útil. Não acrescente mais nenhum texto ou caracter desnecessário 
            apenas a mensagem informando envolvendo algum detalhe do produto usando a descrição acima como guia e a seção que o
            mesmo se encontra.
            E por fim caso a mensagem do usuário: 
            
            {pre_process_prompt(user_prompt)} 
            
            tenha alguma palavra que pareça com lista, por exemplo,
            'list', liste, listo, listw, gere uma possível lista de compras de acordo 
            com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
            a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
            não gere nenhuma lista apenas a mensagem principal.

            """
            
        else:
            answer_prompt = f"""Considerando que o cliente do nosso 
            mercado {supermarket_name} quer saber {user_prompt} 
            que o produto que ele deseja está atualmente indisponível, 
            responda com uma frase simples e útil ao cliente.
            Considere esse texto {generated_detailed_prompt.text} como exemplo e melhore a sua resposta.
        E por fim caso a mensagem do usuário: 
            
            {pre_process_prompt(user_prompt)} 
            
            tenha alguma palavra que pareça com lista, por exemplo,
            'list', liste, listo, listw, gere uma possível lista de compras de acordo 
            com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
            a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
            não gere nenhuma lista apenas a mensagem principal,
            leve em consideração o produto que ele queria que estava indisponível e gere
            uma lista que substitua o produto da melhor forma possível explicando o porque da lista"""

    else:
        answer_prompt = f"""Considerando que o cliente do nosso 
            mercado {supermarket_name} quer saber {user_prompt} 
            responda com uma frase simples e útil ao cliente. 
            E por fim gere alguma evento ou ação para o cliente fazer
            no mercado se vier a o caso, por exemplo, "Queria saber que horas o
            mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
            """
        
    response = model_2.generate_content(answer_prompt)

    global index_dict_answer

    index_dict_answer = index_dict_answer + 1
    name = f"Answer_{index_dict_answer}"

    conversation_history[name] = response.text


    print(response.text)

    while True:
        user_prompt = str(input("Posso ajudar em algo mais? "))
        

        # print(find_product)

        # # if find_product['Disponivel'].bool(): 
        # if find_product['Disponivel'].item(): 
        #     print(True)
        
        # if(user_prompt.lower() in ['sair', 'exit','no', 'nao']):
            # print(process_prompt(user_prompt.lower()))
        #     print('Até logo!')
        #     break

        if process_prompt(user_prompt):
            break

        find_product = generate_and_search(user_prompt)
    
        is_product = isinstance(find_product, pd.DataFrame)

        if is_product:   

            detailed_prompt = f"""
            Gerar uma descrição detalhada do produto {find_product['Nome']} para um chatbot de e-commerce
            """

            generated_detailed_prompt = model_2.generate_content(detailed_prompt)

            if find_product['Disponivel'].item():
                answer_prompt = f"""Considerando que o cliente do nosso
                mercado {supermarket_name} enviou a mensagem:

                {pre_process_prompt(user_prompt)} 

                e o possível produto que ele deseja se encontra na seção {find_product['Secao'].item()}, 
                Considere essa descrição detalhada desse produto: 

                {generated_detailed_prompt.text} 

                utilize-la como um guia para melhorar a resposta para o cliente
                e responda com uma frase simples e útil. Não acrescente mais nenhum texto ou caracter desnecessário 
                apenas a mensagem informando envolvendo algum detalhe do produto usando a descrição acima como guia e a seção que o
                mesmo se encontra.
                E por fim caso a mensagem do usuário: 
                
                {pre_process_prompt(user_prompt)} 
                
                tenha alguma palavra que pareça com lista, por exemplo,
                'list', liste, listo, listw, gere uma possível lista de compras de acordo 
                com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
                a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
                não gere nenhuma lista apenas a mensagem principal.

                """
            
            else:
                answer_prompt = f"""Considerando que o cliente do nosso 
                mercado {supermarket_name} quer saber {user_prompt} 
                que o produto que ele deseja está atualmente indisponível, 
                responda com uma frase simples e útil ao cliente.
                Considere esse texto {generated_detailed_prompt.text} como exemplo e melhore a sua resposta.
                E por fim caso a mensagem do usuário: 
                
                {pre_process_prompt(user_prompt)} 
                
                tenha alguma palavra que pareça com lista, por exemplo,
                'list', liste, listo, listw, gere uma possível lista de compras de acordo 
                com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
                a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
                não gere nenhuma lista apenas a mensagem principal,
                leve em consideração o produto que ele queria que estava indisponível e gere
                uma lista que substitua o produto da melhor forma possível explicando o porque da lista"""

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

        index_dict_answer = index_dict_answer + 1
        name = f"Answer_{index_dict_answer}"

        conversation_history[name] = response.text
        print(f"\nCONVERSATION HISTORY - {conversation_history}\n")

user_interaction()
