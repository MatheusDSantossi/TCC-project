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

from database_connection import conn
import psycopg2

import spacy
import re

import json
import ast

import tkinter as tk
from tkinter import scrolledtext

import time
from prompt_searcher import find_section, find_price, find_similar_products_using_input_embeddings

# Você: quero um refrige


# Hecate: 'Temos refrigerantes Guaraná Antárctica e Soda Limonada, em tamanhos tradicionais e mini, na Seção 3. 😉
# '

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

supermarket_name = "Supermercado Santos"

# According to Greek mythology, Hecate, the goddess 
# of magic, crossroads, and the night, is often depicted 
# as a benevolent deity who helps lost people find 
# their way

chatbot_name = "Hecate"

# My plan is to give the opportunity and way to the client put some information like the ones bellow
info_about_the_business = ["Nosso mercado abre as 7h e fecha as 22h", 
                           "Nosso mercado tem o valor de sempre atender os clientes com o melhor atendimento possível",
                           "O nome do nosso gerente é Fulano"
                           ]

# conversation_history = {}

# index_dict = 0

# index_dict_answer = 0

# dict_of_client_products = {}

pre_processed_prompt_products_list = []

nlp = spacy.load("pt_core_news_lg")

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

# conn.commit()
# # conn.close()

# # 1	"#REFRIG. GUARANA ANTARCT PET CACULINHA"	"Seção 3"	1.2	"PC"	"22021000"	0.89	0.89	1.2	true
# # 2	"REFRIG. GUARANA ANTARCT PET"	"Seção 3"	6	"PC"	"22021000"	4.89	4.89	6	true


# for row in rows:
#     products = {
#         "Nome": row[1],
#         "Secao": row[2],
#         "PrecoVenda": float(row[3]),
#         "UnidadeMedida": row[4],
#         "Disponivel": row[9],
#     }
#     products_list.append(products)


# products_df = pd.DataFrame(products_list)

# # print(products_df)


# df = pd.DataFrame(products_df)
# # df.columns = ['Nome', 'Secao', 'Disponivel']
# df.columns = ['Nome', 'Secao', 'PrecoVenda', 'UnidadeMedida', 'Disponivel']

# # print(df)

# embed_model = 'models/embedding-001'
embed_model = 'models/text-embedding-004'

#    content=f'O produto {title} é encontrado na {secao} com o preço de {preco_venda} reais possuindo a unidade de medida {unidade_medida}',  # Include the type of the product

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

# print(embed_fn("Arroz"))


#df["Embeddings"] = df.apply(lambda row: embed_fn(row["Nome"], row["Secao"], row["PrecoVenda"], row['UnidadeMedida']), axis=1)
# df["Embeddings"] = df.apply(lambda row: embed_fn(row["Nome"]), axis=1)

# print(df)

# Transformar DataFrame em uma lista de tuplas
# data_tuples = [tuple(x) for x in df.to_numpy()]

# # print(data_tuples)

# sql = """INSERT INTO products_with_embeddings(
#     nome,
#     secao_produto,
#     preco_venda,
#     unidade_medida,
#     disponivel,
#     embeddings
# ) 
# values(%s, %s, %s, %s, %s, %s) RETURNING *"""

# try:
#     with conn.cursor() as cur:
#         for record in data_tuples:
#             cur.execute(cur.mogrify(sql, record))

#     conn.commit()
# except (Exception, psycopg2.DatabaseError) as error:
#     print(error)
# finally:
#     conn.close()

## NOW I HAVE THE DF IN MY DATABASE

sql = "SELECT * FROM products_with_embeddings"

cur = conn.cursor()

cur.execute(sql)

rows = cur.fetchall()

conn.commit()

# conn.close()

products_list = []

# 205	"SAB.PROTEX ERVA DOCE EMB 1 X 12 90G"	"Seção 2"	
# 2.0	"PC"	true	{-0.0068901023,-0.017110113,-0.04398383,-0.019743577,0.038723197,0.0059666275,-0.026354691,-0.010065305,-0.008089105,0.05907605,0.0034906971,-0.0032800054,0.01664473,-0.01100858,-0.032542452,-0.024404066,0.02496846,0.031474207,-0.022154966,-0.047623083,0.009256254,0.018447643,0.0070926216,0.009886847,-0.01581496,-0.006393073,0.035618663,-0.056431677,-0.034084078,0.0153410295,-0.0874218,0.068573475,-0.0711557,-0.044712923,-0.05187479,-0.006223573,-0.025455125,-0.004621894,-0.027902637,0.023120308,0.0030681025,0.0029099355,0.0054100836,-0.037552994,0.02561197,0.024048232,0.0076997364,-0.024422921,0.021032508,-0.07029078,0.0342872,-0.00075353414,0.054282833,0.014486551,-0.016163412,-0.036995027,0.008775744,-0.018168617,0.019449182,-0.011274251,-0.03281642,0.019826185,0.040396605,0.054529514,-0.022716627,-0.04045984,-0.031299558,0.012925805,0.06933866,-0.024309542,0.007416687,-0.015563377,0.024038836,-0.015188449,-0.0050065126,-0.038519662,-0.03186893,0.097649105,0.04311643,0.043365747,0.012323822,-0.028255459,-0.04534822,-0.046334583,0.0015166552,0.046734765,-0.04296521,-0.054640185,0.007004131,0.047138255,0.028350465,-0.059278913,0.012525092,-0.060510036,0.012723434,0.07966275,-0.035475496,-0.053003635,-0.012922144,-0.07064509,-0.029028114,-0.002136717,-0.037108675,-0.022945497,-0.018494166,-0.034813542,-0.017104164,0.033728227,-0.043474164,0.04273438,-0.020207595,-0.024624173,0.03435385,0.002797935,0.022001414,-0.01860624,0.02521641,0.050557002,0.07429551,0.040560704,0.025019515,-0.005631261,0.03583172,0.040830955,0.021474002,-0.019694611,-0.007966408,0.054567456,0.045272112,-0.00023993853,-0.021013565,0.018896857,0.0032448154,0.024153566,0.05847691,0.074323796,0.046152443,0.038300924,-0.016214319,0.028515749,-0.00234592,0.06029181,0.011173096,0.0025296675,-0.017160915,0.029064033,0.008516131,0.0508219,0.045304593,-0.009537978,0.0036333883,-0.006842084,-0.07063358,0.04460528,0.07200772,0.011467511,-0.0023403757,-0.022922864,0.008469214,0.0137980245,0.07707153,-0.016193608,-0.019483523,0.012631832,-0.009324361,-0.06274002,0.00010001747,0.02120844,0.01109925,0.009890586,0.03540926,0.0126273865,-0.046941776,-0.0328562,-0.04063749,-0.059650183,-0.01916512,-0.015786037,-0.06456854,-0.010441197,-0.03969247,-0.055640016,0.021796392,0.028445158,0.008600869,0.0017804515,0.080680475,-0.043215036,0.0016222637,0.013587025,-0.0060471524,-0.032888535,-0.08759772,0.039177816,0.010283359,0.02481573,-3.0994983e-05,0.006613448,0.019050987,-0.025407178,-0.036984924,0.030352226,-0.0076094475,0.014928605,8.713977e-05,-0.038096484,0.023603082,-0.039640985,-0.026583081,0.029287105,-0.04305839,0.0030001746,0.013633362,-0.010077867,0.051654566,-0.0047395136,0.029988073,-0.02059836,-0.035130233,-0.010733176,-0.021058114,-0.03674546,-0.016719684,0.021591373,-0.058107592,0.05467076,-0.063229255,0.013544913,0.023141867,-0.069302715,-0.027010685,0.06130083,0.028064672,-0.0031814915,0.05441555,-0.03250148,0.055368975,0.051454403,0.028632095,0.020543886,0.007422717,0.055679917,-0.026338026,0.033912078,-0.03662622,-0.029654041,-0.029974418,0.07534687,0.0021882828,0.06287127,-0.006224373,-0.04251881,-0.010882219,-0.0175953,-0.06939704,-0.004320797,-0.080255635,0.0051224898,0.036000483,0.00064698665,0.07778502,-0.00026889183,-0.0011103213,-0.02921944,-0.01430382,-0.0121785635,0.040554978,-0.031660393,-0.028372694,0.04767138,0.010363762,-0.0037805191,0.05085381,0.026511272,0.027814321,-0.003162316,-0.029558552,0.08522062,0.027492581,-0.072494425,-0.0008790665,0.013428953,0.013328018,-0.042419028,-0.018424068,-0.0075674606,-0.038437758,-0.065055996,0.04592963,-0.0031968236,-0.02156277,-0.032736547,-0.033172715,0.004969252,0.0032323068,-0.00058135943,-0.018817445,0.045015734,0.018790705,-0.007868373,0.041108076,-0.05547537,0.019731883,-0.060015798,0.031261567,0.0022004615,0.013123242,-0.03353897,0.014208934,0.018529871,0.026389424,-0.023826541,-0.033223733,0.026245514,0.033224776,0.0010385073,-0.030177385,0.024835743,-0.029030243,0.039087843,0.007907275,0.113444395,0.047353618,0.022472437,-0.005092387,0.017935636,-0.04765569,0.05553472,-0.033002853,0.014157902,-0.021556105,0.0025807875,-0.015075465,0.06254213,-0.013185628,-0.026598796,-0.06698265,-0.013689384,0.00028969932,0.002891159,0.010998672,0.028213426,-0.038055286,-0.030936195,0.044435583,0.030971242,-0.039587412,0.032047722,0.0014350354,0.052762695,0.027921557,0.030869428,-0.0015409018,0.019527046,0.019008083,-0.038919277,0.03787943,-0.016027793,-0.007839852,-0.0237089,-0.085783854,0.032306176,-0.042824384,-0.006464784,0.013920692,0.028777761,0.021475105,0.02464172,-0.03658259,0.064592935,0.025575636,-0.019270748,0.06099152,0.02278317,-0.007005583,-0.0070437808,0.0042988583,-0.0121412305,0.011709634,0.025521364,-0.042219892,-0.07180534,0.035863917,-0.015476906,-0.005156069,0.0071871397,0.058992576,0.023095695,0.035260726,0.027871622,-0.010307579,0.0011459253,0.08923387,0.039672088,-0.008752804,-0.0050207954,-0.07203241,-0.03928028,0.033774137,-0.0077734776,-0.016135668,-0.027514836,-0.032666784,-0.005059445,-0.07975377,0.002535732,0.0005188402,-0.087696865,-0.058101017,-0.020956254,-0.01006966,0.030014237,0.02294807,-0.044227347,-0.058195617,-0.022685854,0.066295326,-0.02425535,0.006694341,0.019963387,0.007419015,0.01564112,-0.005974731,-0.009422186,-0.06750804,-0.0037738862,0.051998336,-0.03720036,0.013552176,0.022472434,0.041915216,-0.04580574,0.07453997,-0.04551681,-0.007835532,-0.029369468,0.026948694,0.02643494,-4.0998024e-05,-0.008016646,0.0042766426,0.03834745,0.022449395,0.005899985,-0.04569184,-0.0030291216,0.0053872955,0.0192113,0.024949914,-0.08734741,0.009815149,-0.06667957,0.0064982753,-0.05372912,0.0093537485,0.01830826,0.0040839566,0.05888066,0.01972405,0.017985394,-0.016488558,-0.02103893,-0.007163879,-0.05674633,0.045777895,0.0072280057,0.009957965,0.018332027,0.016361112,0.04890819,0.016135208,-0.052089844,0.020017771,0.00564552,-0.02062612,-0.008045862,-0.06625649,0.021861635,-0.021348206,-0.036585156,-0.046021633,-0.032877527,0.0048297164,-0.011350187,0.014378888,0.01946808,-0.026161656,0.047111087,0.053959385,0.0470423,-0.049267195,-0.026195027,-0.06979532,-0.08138142,0.02214683,0.033137687,-0.07896152,0.0033881704,-0.041859474,0.037162814,-0.009467815,0.024729319,-0.0038370646,0.043933615,-0.013555603,-0.049048834,0.070332415,0.06801688,-0.0083442945,-0.0027902045,0.027513348,0.0026978946,-0.05008742,0.057609066,0.034506217,-0.020999506,-0.022601066,-0.016484369,0.018100647,-0.040746458,-0.008121214,0.004383581,-0.07589758,-0.012050501,0.04480719,-0.05304751,-0.013690901,0.038829423,-0.078854874,0.07949355,-0.021344716,0.08041059,-0.060053382,0.0051456452,-0.0027721126,0.0068186535,0.024351185,0.057489906,0.03767477,-0.005147226,0.0024537165,-0.04167769,0.014959994,0.04152857,-0.049202163,-0.02185216,0.022262184,-0.0526753,0.018319631,-0.004930662,0.0049799373,0.04044112,0.014398871,-0.063585944,0.009708782,-0.025556168,-0.0406501,0.0035988865,0.02574285,-0.05654094,0.00032588528,-0.031255685,-0.018485311,-0.017045714,0.0629155,0.007378451,-0.013917539,-0.016934963,0.030927934,-0.028127726,0.039349977,0.04291419,0.017114747,-0.010473358,-0.014307922,-0.013648086,-0.03870527,0.0017305054,-0.046343345,-0.050375305,0.055358414,-0.033092923,0.034786154,0.039539088,-0.0005971536,0.0068878215,0.012631144,0.031436216,-0.013454105,0.034155447,-0.008708155,0.060785566,-0.032157708,0.022649068,-0.026518062,-0.03332039,-0.018273076,-0.043418877,0.012099797,-0.08159079,0.017102636,-0.026242053,0.0694269,0.011372458,0.040214416,0.0039879344,0.0009584602,-0.003705355,-0.01355495,0.026294751,-0.019167572,-0.05169133,0.0050167493,0.006931846,-0.040880375,-0.01570934,0.033269055,-0.0053437506,-0.02120884,-0.028605316,-0.005681296,0.014233413,-0.012084472,0.014398712,-0.006100458,0.0031594664,-0.008224307,-0.09771763,0.059706792,-0.009810868,0.040582385,0.05892565,-0.038271997,0.036377307,-0.0076692738,0.02643679,-0.015749808,0.015358362,0.011385062,0.010645426,-0.07730976,-0.022527937,0.032514744,0.02973783,-0.05439423,0.08050984,0.006747693,-0.08386579,-0.105559774,0.00371902,0.027535135,-0.03323133,0.019598199,0.021959908,0.018142885,-0.0062465817,-0.032821033,0.0044373074,0.006653121,-0.049082763,-0.054879345,0.038442563,0.011343819,0.037066147,-0.01630943,-0.00019665304,-0.033951353,-0.06311724,-0.008253971,0.04531699,-0.06266875,0.014882386,0.08993492,-0.007617792,0.06489596,0.020573858,-0.07149441,0.021619506,-0.032685746,0.015063036,-0.013628743,-0.016095899,-0.031375688,-0.006214086,-0.06531222,0.03806269,0.039781965,-0.024895867,-0.032784887,-0.0030122967,0.006966487,-0.021398194,0.0023490994,0.019846719,0.037059497,0.017099291,0.026226927,0.013568375,-0.044594254,0.034976944,-0.018595597,0.0062386105,0.022447603,0.004695133,-0.0367035,-0.03207425,-0.009584113,0.0054462967,0.016598439,0.008393014,0.016650172,-0.055286463,-0.05693451,0.06550256,0.037636433,-0.08449789,0.039674863,0.013158433,-0.035836704,0.029676555,0.031551186,0.020272214,0.0021283668,-0.0057775085,-0.012030719,-0.02279491,-0.008728122,-0.013957174,-0.041581154,0.019986734,0.08891604,0.025756853,-0.02714344,-0.013661843,-0.042035628,0.03641661,-0.03606498,0.018397596,-0.016161961,-0.043704353,-0.0973365,-0.008373395,-0.0076079126,0.05033822,0.031660218,-0.0112589765,-0.027697546,-0.047674447,-0.022257691,-0.050725274,-0.016216947,0.0068274955,0.016117826,0.030916812,0.06056305,0.026924703,0.017211309,-0.005477697,0.0034638282,0.052799866,0.01697508,0.03388141,-0.009928845,-0.005880076,-0.0034014771,0.04829151,-0.014804046,-0.0075884615}


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
products_available = [p["Nome"] for p in products_df.to_dict(orient='records') if p['Disponivel']]

def for_dividing_prompt_limit():
    products_list_2 = []
    
    for p in products_df.to_dict(orient='records'):
        # prompt += p['Nome']
        products_list_2.append(p['Nome'])
        
    max_prompt_size = 425
    
    for i in range(0, max_prompt_size):
        prompt+= products_list_2[i]

# Process the prompt using NLTK 
def process_prompt(prompt):
    """
    This function processes the user's prompt to check if the user wants to exit.

    # return prompt_representation#-
    Parameters:
    prompt (str): The user's input prompt.

    Returns:
    bool: True if the user wants to exit, False otherwise.
    """
    
    print("PROMPT: ", prompt)
    if fuzz.token_set_ratio(prompt.lower(), "quero sair") >= 80:
        # print(process_prompt(user_prompt.lower()))  # Processa antes de sair
        print(f'Obrigado por vir ao {supermarket_name}. Volte sempre!')
        return True
    return False

def pre_process_prompt_grammatical(prompt):
    
    # OBS: The input INSET is being blocked for some reason with the error message:
    # ValueError: Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`, but none 
    # were returned. Please check the `candidate.safety_ratings` to determine if the response was blocked.
    
    fix_prompt = f"""
        Corrija os erros gramaticais e ortográficos no seguinte texto para que ele seja compreensível para um leitor 
        humano e normalize os nomes para uma forma padrão, seja eles produtos, utensilíos ou qualquer outra coisa. Leve em consideração que 
        esse texto está sendo retirado do cliente do estabelecimento {supermarket_name} leve isso em consideração para corrigir da melhor
        forma possível. Considerando que as palavras podem aparecer abreviadas como "INSET SBP AERO" que significa "INSETICIDA SBP AEROSSOL" 
        faça uma pesquisa aprofundada do que cada palavra pode significada estando abreviada ou não, considerando que muito provavelmente todas 
        elas são produtos de varejo encontrados em qualquer supermercado, pense nisso também quando for tentar identificar as palvras. Segue o texto para ser corrigido:

        Texto:

        "{prompt}"

        Mantenha o significado original do texto e corrija apenas os erros de escrita e gramática. Por exemplo, "quero um cafezinho" 
        deve ser corrigido para "Quero um cafe", "Que dea é heij" para "Que dia é hoje", "Que horas se" para "Que horas são",
        "Sarveti" para "Sorvete", "Seai" para "Saia", preste bastante atençaão para as nuances do texto, identifique e corrija os erros
        presente nele. Outros exemplos de possíveis correções: 
        "Quanto custa o bisc e o sab" para "Quanto custa o biscotio e o sabão" ou "Quanto custa o biscotio e o sabonete"
        "Aendo figa localezado os tomate e a cebole" para "Aonde fica localizado os Tomates e a Cebola", "Vocês te jel, deterge e saco" para "Vocês têm gel, detergente e suco"
        Preste bastante atenção aos minimos detalhes do texto, perceba que alguns nomes podem está completamente incorretos, abreviados ou incompletos como "refrig.", "q h o mercado abre" que teria como correção "que horas o mercado abre" 
        leve isso em consideração antes de fazer a correção do texto. Faça isso para qualquer palavra minimamente errada. Porém, não adicione texto algum ou produtos
        que não tem no texto. Seja o mais direto possível, sem acrescentar informações adicionais. 

        Regras de Retorno:

        Caso o texto não contenha nenhum produto ou seja, o texto disponibilizado é algo totalmente diferente a Lista de produtos que possam ser vendidos no {supermarket_name}, como por exemplo,
        "Ole, como você está?", "Quera saber que horas o mercado facha?", "Como é o seu nome", "Qual o estado do mercado" ou "Como funciona o mercado" 
        corrija apenas os erros gramaticais e retorne o texto.
    """
    # In case the product is not in the list, show the message "Produto não está disponível!"
    
    time.sleep(3)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt)

    # Problema agora é identificar mais de um produto :)

    return pre_processed_prompt.text

def fetch_products_alike(product_name):
    with conn.cursor() as cursor:
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()
        
        query = """
            SELECT nome, secao_produto, preco_venda FROM products_with_embeddings
            WHERE SIMILARITY(nome, %s) > 0.1;
        """
        
        # print(type(product_name))
        
        product_dict = {}
        
        # for keyword in product_name:
        cursor.execute(query, (product_name,))
        products = cursor.fetchall()
        conn.commit()
            
        for product in products:
            if product[0] not in product_dict:
                product_dict[product[0]] = []
            product_dict[product[0]].extend(product[1].split(","))
            
    return product_dict

def pre_process_prompt(prompt):
    
    fix_prompt_v1 = pre_process_prompt_grammatical(prompt)
    print(f"FIXED 1 {fix_prompt_v1}")
    
    products_alike = fetch_products_alike(fix_prompt_v1)

    # print("PRODUCTS ALIKE: ", products_alike)
    
    input_embed = embed_fn(fix_prompt_v1)
    
    print("INPUT EMBED: ", input_embed)
    
    products_find_in_section = find_section(fix_prompt_v1, conn)
    products_find_with_price = find_price(fix_prompt_v1, conn)
    products_find_using_prompt = find_similar_products_using_input_embeddings(input_embed, conn, np, cosine_similarity)
    
    if(products_find_in_section != None):
        fix_prompt_v2 = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        """
    elif products_find_with_price != None:
        fix_prompt_v2 = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.

        Provided Information:

        1. Products with the same price:
        {products_find_with_price}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"
        
        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        """
    else:
        fix_prompt_v2 = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.

        Provided Information:

        1. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        """
    
    time.sleep(3) # a little sleep to avoid overflow of questions to Gemini API
    
    print("FIX_PROMP_V2: ", fix_prompt_v2)
    
    pre_processed_prompt = generate_content_model.generate_content(fix_prompt_v2)

    return pre_processed_prompt.text

# pre_process_prompt("quais produtos custam 2.30")

# pre_process_prompt("Quanto custa o café")


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


    if not process_prompt(prompt):
        

        fix_response = pre_process_prompt(prompt)

        prompt = fix_response
        
        print("CHATBOT RETURN: ", prompt)
        
        # global index_dict 
        # index_dict += 1
        # name = f"Question_{index_dict}"
        # conversation_history[name] = prompt
        
        new_prompt = prompt.lower()
        new_prompt_without_spaces = new_prompt.strip()

        # new_prompt = prompt.strip()
        
        # new_prompt = unicodedata.normalize('NFKD', prompt).encode('ascii', 'ignore').decode('ascii')
        
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
        
        
        if (new_prompt_without_spaces not in ['produto na', 'texto ra']):
            print("new_prompt: ", new_prompt_without_spaces)
            # generate_synonymous(pre_processed_prompt_products_list)

            contains_a_product = True

            # Using TF-IDF to text comparation
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform([new_prompt] + products_df['Nome'].tolist())  

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
                
                Informações sobre o {supermarket_name}:\n
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
            conn.commit()   
    



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
    #         conn.commit()
            
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
    #         conn.commit()
            
    #     except (Exception, psycopg2.DatabaseError) as error:
    #         print(error)
            
    #     # finally:
    #     #     conn.close()           
            

# user_interaction()

# um algoritmo que descreve a inclusão de um novo topico



def chatbot_response(user_prompt):
    
    global dict_of_client_products
    global products_list_after_user_input

    find_product = generate_and_search(user_prompt)

    is_product = isinstance(find_product, pd.DataFrame)
    
    print("USER PROMPT: ", user_prompt)
    
    if process_prompt(user_prompt):
        return

    else: 
        
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
                    
                for product in products_list_after_user_input: 
                    detailed_prompt += f"Nome do produto: {product['Name']}\n"
                    detailed_prompt += f"Preço do produto: {product['Price']}\n"
                    detailed_prompt += f"Produto está disponível (False=Não, True=Sim): {product['Available']}\n\n"
                
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
                mercado {supermarket_name} enviou a mensagem:

                {pre_process_prompt_grammatical(user_prompt)} 

                e os possíveis produtos que ele deseja são esses 
                
                """
                # for product in pre_processed_prompt_products_list: 
                for product in products_list_after_user_input: 
                    answer_prompt += f"Nome do produto: {product['Name']}\n"
                    answer_prompt += f"Preço do produto: {product['Price']}\n"
                    answer_prompt += f"Seção do produto: {product['Section']}\n"
                    answer_prompt += f"Produto está disponível (False=Não, True=Sim): {product['Available']}\n\n"

                # {pre_processed_prompt_products_list}.
                answer_prompt += f"""
                Considere também essa descrição detalhada dos produtos: 

                {generated_detailed_prompt.text} 

                utilize-la como um guia para melhorar a resposta para o cliente
                e responda com uma frase simples e útil levando em conta todos os produtos e informações disponibilizadas. Não acrescente mais nenhum texto ou caracter desnecessário 
                apenas a mensagem informando envolvendo algum detalhe do produto usando a descrição acima como guia e a seção que o
                mesmo se encontra.
                
                Além dessas informações sobre o {supermarket_name}:
                """
                
                for info in info_about_the_business:
                    print(f"INFO: {info}")
                    answer_prompt+=f"{info}\n"
                
                answer_prompt+=f"""
                E por fim caso a mensagem do usuário: 
                
                {pre_process_prompt_grammatical(user_prompt)} 
                
                tenha alguma palavra que pareça com lista, por exemplo,
                'list', liste, listo, listw, gere uma possível lista de compras de acordo 
                com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
                a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
                não gere nenhuma lista apenas a mensagem principal.

                """
                
            else:
                answer_prompt = f"""Considerando que o cliente do nosso 
                mercado {supermarket_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
                que o produto que ele deseja está atualmente indisponível, 
                responda com uma frase simples e útil ao cliente.
                Considere esse texto {generated_detailed_prompt.text} como exemplo e melhore a sua resposta.
                E por fim caso a mensagem do usuário: 
                
                {pre_process_prompt_grammatical(user_prompt)} 
                
                tenha alguma palavra que pareça com lista, por exemplo,
                'list', liste, listo, listw, gere uma possível lista de compras de acordo 
                com os produtos disponíveis no mercado que são esses, {products_available}. Repetindo, caso se e somente se
                a mensagem do usuário contenha alguma palavra que remeta a lista, caso não tenha
                não gere nenhuma lista apenas a mensagem principal,
                leve em consideração o produto que ele queria que estava indisponível e gere
                uma lista que substitua o produto da melhor forma possível explicando o porque da lista"""

        else:
            
            products_alike = fetch_products_alike(user_prompt)
            
            if user_prompt == "Saber mais sobre os produtos":
                answer_prompt = f"""
                    Considerando que o cliente do nosso
                    mercado {supermarket_name} enviou a mensagem:

                    {user_prompt} 

                    e os possíveis produtos temos disponíveis são esses: 
                    \n
                    """
                    
                with conn.cursor() as cursor:
                    
                    sql_query = """
                    SELECT nome, secao_produto, preco_venda, disponivel FROM products_with_embeddings
                    ORDER BY RANDOM() LIMIT 20;
                    """
                    
                    cursor.execute(sql_query)
                    result = cursor.fetchall()
                    
                    conn.commit()
                
                product_info = ""
                
                for product in result:
                    for i in range(len(product)):
                        # print("RANDOM PRODUCT: ", product[i])
                        product_info += str(product[i]) + " " 
                    
                    answer_prompt+=product_info.strip() + "\n"
                
                answer_prompt+="""
                Considerando esses produtos que contém os seguintes valores nome do produto, seção do produto, preco de venda e se o mesmo está disponivel 
                extamente nessa sequência, retorne uma mensagem com alguns produtos que temos para ver se o cliente vai querer algum deles
                retorne a melhor resposta possível.
                """
                
            elif user_prompt == "Saber mais sobre nosso mercado":
                answer_prompt = f"""
                    Considerando que o cliente do nosso
                    mercado {supermarket_name} enviou a mensagem:

                    {user_prompt} 

                    e as informações sobre o {supermarket_name} que temos disponíveis são essas: 
                    \n
                    """
                    
                for info in info_about_the_business:
                    answer_prompt+=f"{info}\n"
                
                answer_prompt+="""
                Considerando essas informações retorne uma mensagem com elas e alguma
                possível ação ou pergunta que o cliente pode fazer agora.
                retorne a melhor resposta possível.
                """
                
            else:
                
                answer_prompt = f"""Considerando que o cliente do nosso 
                    mercado {supermarket_name} quer saber {pre_process_prompt_grammatical(user_prompt)} 
                    responda com uma frase simples e útil ao cliente. 
                    E por fim gere alguma evento ou ação para o cliente fazer
                    no mercado se vier a o caso, por exemplo, "Queria saber que horas o
                    mercado abre?", resposta, "O mercado abre as 7h e hoje temos uma grande promoção"
                    
                    Leve em consideração essas informações sobre o {supermarket_name}:
                    """
                
                for info in info_about_the_business:
                    answer_prompt+=f"{info}\n"
            

        # response = main_chatbot_model.generate_content(answer_prompt)
        response = chat.send_message(answer_prompt)

        print(f"ANSWER PROMPT {answer_prompt}")

        # print(response.text)

        # index_dict_answer = index_dict_answer + 1
        # name = f"Answer_{index_dict_answer}"

        # conversation_history[name] = response.text
        # print(f"\nCONVERSATION HISTORY - {conversation_history}\n")
        
        # I changed the Gemini chatbot model and now he decorates
        # the conversation history automatically, so it's unnecessary
        # to keep the conversation history in my DB
        sql = """INSERT INTO conversation_history (
            user_input,
            chatbot_response)
            VALUES (%s, %s) RETURNING *"""
        
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (pre_process_prompt_grammatical(user_prompt), response.text))
            conn.commit()
            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        
        print("-"*80)
        print("RESPONSE: ", response.text)
        return f"'{response.text}'"


chatbot_response("cafe")

def generate_synonymous(product=None, product_list=None):
    
    print("product: ", product)
    original_product = product
    
    # semantic_dictorinary = create_semantic_dictionary(product)
    products = generate_and_search(product)
    
    products_dict = {}
    product_synonymous_list = []
    
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
    
    # print("""PRODUCTS: """, products)
    
    # if not product_list:
    if not products_dict:
    
        new_product = pre_process_prompt_grammatical(product)
        
        prompt = """Coloque todos os sinônimos possíveis para as palavras abaixo. Leve em
                    consideração que as palavras podem aparecer abreviadas como "INSET SBP AERO" que 
                    significa "INSETICIDA SBP AEROSSOL" faça uma pesquisa aprofundada do que cada palavra
                    pode significada estando abreviada ou não, considerando que muito provavelmente todas 
                    elas são produtos de varejo encontrados em qualquer supermercado, pense nisso também quando
                    for tentar identificar as palvras. Não repita os sinônimos e se limita a no máximo 30 sinônimos.
                    Não adicione nenhum texto ou caracter extra como por exemplo  ```python, ```json  ou ```markdown
                    ou algo parecido retorne apenas os sinonimos. 
                    Segue o/os produtos.
        \n"""
        
        prompt+=new_product
        
        prompt += """\n\nRetorne todos os produtos e sinônimos como um dicionário, por exemplo: {'CAFE PILAO ALMOFADA': ['Expresso', 'Cappuccino', 'Mocha', 'Latte', 'Café da Manhã', 'Café Coado', 'Café em Grãos']}
        lembre-se de abrir a chave { e fechar a mesma chave } retorne apenas o diciónario sem nenhum texto adicional, 
        ou seja, a sua resposta deve começar com uma { e terminar com o fechamento dela }. Adicione todos os possíveis sinônimos de cada palavra
        Lembre de colocar as palavras entre aspas duplas, por exemplo, {"Caixa de Fósforos","Palito de Fósforo","Fósforo de Segurança"}
        """ 
    else:
        prompt = """Coloque todos os sinônimos possíveis para as palavras abaixo. Leve em
                    consideração que as palavras podem aparecer abreviadas como "INSET SBP AERO" que 
                    significa "INSETICIDA SBP AEROSSOL" faça uma pesquisa aprofundada do que cada palavra
                    pode significada estando abreviada ou não, considerando que muito provavelmente todas 
                    elas são produtos de varejo encontrados em qualquer supermercado, pense nisso também quando
                    for tentar identificar as palavras. Não adicione nenhum texto ou caracter extra como por exemplo  ```python, ```json  ou ```markdown
                    ou algo parecido retorne apenas os sinonimos. 
                    Segue o/os produtos.
                \n"""
        
        for product in product_synonymous_list:
            prompt += str(product["Name"]) + "\n"
        
        prompt += """\n\nRetorne todos os produtos e sinônimos como um dicionário, por exemplo: {'CAFE PILAO ALMOFADA': ['Expresso', 'Cappuccino', 'Mocha', 'Latte', 'Café da Manhã', 'Café Coado', 'Café em Grãos']}
        lembre-se de abrir a chave { e fechar a mesma chave } retorne apenas o diciónario sem nenhum texto adicional, 
        ou seja, a sua resposta deve começar com uma { e terminar com o fechamento dela }. Em formato de dicionário em Python
        não retorne em nenhum outro formato.
        Em hipotese alguma adicione nenhum texto ou caracter extra como por exemplo  ```python, ```json  ou ```markdown
        ou algo parecido retorne apenas os sinonimos. 
        Adicione todos os possíveis sinônimos de cada palavra
        Lembre de colocar as palavras entre aspas duplas, por exemplo, {"Caixa de Fósforos","Palito de Fósforo","Fósforo de Segurança"}
        """ 

    
    print("GENERATE SYNONYMOUS PROMPT: ", prompt)
    print("==========================================")
    
    # print("PROMPT:", prompt)   
    response = generate_content_model.generate_content(prompt)
    print(response.text)
    
    # {'Macarrão Pilsen Médio': ['Macarrão', 'Massa', 
    #                            'Espaguete', 'Talharim', 'Fideli', 'Ninhos', 'Penne', 'Farfalle', 'Spaghetti', 'Macarrão Tipo Pilsen', 'Macarrão Médio', 'Macarrão de Trigo', 'Macarrão Seco', 'Macarrão Cozido', 'Macarrão para Sopa', 'Macarrão para Salada', 'Macarrão para Massa']}
    
    synonymous_dict = ast.literal_eval(response.text)
    # synonymous_list = list(synonymous_dict.values())[0]
    
    first_key = next(iter(synonymous_dict))
    
    synonymous_list = synonymous_dict[first_key]
    # synonymous_dict = dict(synonymous_dict.values())[0]
    print("Synonymous list: ",synonymous_list)
    print("Synonymous dict: ",synonymous_dict)
    
    # for key, value in synonymous_dict.items():
        # print("synonymous key: ",key, "synonymous value: ", value)
        # synonymous_list = synonymous_dict[key]
    
    # Transform list into the format you want
    formatted_string = '{' + ', '.join(f'"{syn}"' for syn in synonymous_list) + '}'
    
    # synonymous_dict_wrapped = f"'{formatted_string}'"
    # print("Synonymous dict wrapped: ",synonymous_dict_wrapped)
    
    # Escaping special characters in the list and converting it to PostgreSQL's array format
    # synonymous_list_pg = '{' + ','.join('"' + syn + '"' for syn in synonymous_list) + '}'

    if isinstance(product, dict):
        print("product: ", product['Name'])
        product = product['Name']


    with conn.cursor() as cursor:
        cursor.execute("""
                       UPDATE products_with_embeddings
                       SET synonymous = %s
                       WHERE nome = %s
                       """, (formatted_string, original_product))
        conn.commit()
            
    return synonymous_list
    
    # text = response.text
    
    # regex = re.compile(pattern_for_syn, text, re.DOTALL)
    
    # matches = re.findall(pattern_for_syn, text, re.DOTALL)
    
    # for match in matches:
    #     print("MATCH: ", match)
    
    # match = re.search(pattern_for_syn, text)

    # if match:
    #     # extracting the correspondent group to the pattern
    #     product_string = match.group(1)
    #     print("PRODUCT STRING: ", product_string)

    #     pre_processed_prompt_products_list = [product.strip() for product in product_string.split(',')]
    
    # start = fixed_text.find('{')+1
    # end =fixed_text.find('}')
    # json_string = fixed_text[start:end]
    
    # dictionary = json.loads(json_string)
    
    # print("==========================================")
    
    # print(pre_processed_prompt_products_list) 





# generate_synonymous("NESCAU 2.0 ACTIGEN-E NESTLE CX COM 2")



# chatbot_response("Qual o noem do mercado de gvoces")

# def send_message():
#     # user_input = user_entry.get()
#     # chat_window.insert(tk.END, f"Você: {user_input}\n")
#     # user_entry.delete(0, tk.END)
    
#     # # Get answer from chatbot
#     # response = chatbot_response(user_input)
#     # chat_window.insert(tk.END, f"{response}\n")

#     user_input = user_entry.get()
#     chat_window.insert(tk.END, f"Você: ", "user_bold")
#     chat_window.insert(tk.END, f"{user_input}\n\n", "user")
#     user_entry.delete(0, tk.END)
    
#     chat_window.insert(tk.END, f"Chatbot está pensando...\n\n", "bot_bold")
    
#     print(f"USER INPUT: {user_input}\n")
    
#     window.after(100, generate_response, user_input)
    

# def generate_response(user_input):
#     chat_window_content = chat_window.get("1.0", tk.END)
#     chat_window_content = chat_window_content.replace("Chatbot está pensando...\n\n", "") 
#     chat_window.delete("1.0", tk.END)
#     chat_window.insert(tk.END, chat_window_content)

#     response = chatbot_response(user_input)

#     chat_window.insert(tk.END, f"{chatbot_name}: ", "bot_bold")
#     chat_window.insert(tk.END, f"{response}\n\n", "bot")
    
# # creating the main window
# window = tk.Tk()
# window.title(f"{supermarket_name}")

# chat_window = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=20)
# chat_window.pack(padx=10, pady=10)

# # Defining the tags for formatation
# chat_window.tag_configure("user_bold", font=("Helvetica", 10, "bold"), justify="right", spacing3=10)
# chat_window.tag_configure("user", justify="right", spacing3=10)
# chat_window.tag_configure("bot_bold", font=("Helvetica", 10, "bold"), justify="left", spacing3=10)
# chat_window.tag_configure("bot", justify="left", spacing3=10)

# chat_window.insert(tk.END, f"{chatbot_name}: ", "bot_bold")
# chat_window.insert(tk.END, f"Olá, me chamo {chatbot_name}. Em que posso lhe ajudar hoje?\n\n", "bot")

# # Entrance of text for the user

# user_entry = tk.Entry(window, width=50)
# user_entry.pack(padx=10, pady=10)

# # Button to send the message
# send_button = tk.Button(window, text="Enviar", command=send_message)
# send_button.pack(padx=10, pady=10)
    
# # Start the interface
# window.mainloop()

# # try:
# #     with conn.cursor() as cur:
# #         for record in data_tuples:
# #             cur.execute(cur.mogrify(sql, record))

# #     conn.commit()
# # except (Exception, psycopg2.DatabaseError) as error:
# #     print(error)
# # finally:
# #     conn.close()