from pathlib import Path
import sys
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure the root directory is included
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import psycopg2
from config.database_connection import conn

# Load known brands if available
KNOWN_BRANDS = [ "ANTARCT", "ANTARC", "LOPES", "CAMIL", "FUGINI", "BABY ROGER", "OUROLUX", "AMAFIL", "SORA", "YPE", "PILAO", "BEM TE VI", "PURO SABOR", "GAROTO", "BAYGON", "HELLMANNS", "ASSOLAN", "PALACIO", "COQUEIRO", "POPULAR", "VITAMASSA", "TOBOGA", "BRILUX", "OMO", "POLITRIZ", "ARCOVERDE", "BETANIA", "CAMPONESA", "PRETINHO", "NESTLE", "VITARELLA", "PRIMOR", "DELICIA", "FORTALEZA", "EVEN", "LUX", "PALMOLIVE", "PHEBO", "TANG", "ALMA FLORES", "SENADOR", "ROSA BRANCA", "SEDA", "NIVEA", "LEGO", "KIT KAT", "DOLCE GUSTO", "NUTELLA", "COCO LÁ", "COSMOPOLITAN", "L'OREAL", "SABO", "WONDER", "TRES COROAS", "LEITE COOP", "CERELAC", "FRUTIGRA", "LIVIA", "CALDENTE", "SABORINHO", "ARAGOA", "PANAMERICANA", "PENFOLDS", "SABO" "COSIPA", "BRASIL CHEESE", "WONDER", "CRESIL", "PANAMERICANA", "COSIPA" "SABO" "CRESIL", "COSIPA", "ARAGOA" "CRESIL", "COSIPA", "PANAMERICANA", "REXONA", "SUAVE", "BARUEL" "STA CLARA", "SANTA CLARA", "MARATA", "LACTA", "CAPRICE", "COLGATE", "TUCS", "DOVE", "CLEAR", "SKALA", "NATHALYA", "TIXAN", "PETINHO", "MILI", "PRIVAX", "YOKI", "FRISCO", "FREEGELLS", "EMARE", "NUTRIMEL", "ARCOR", "TRESEMME", "TABU", "P&G", "SOFRUTA", "PILAR", "SBP", "EMBARE", "SUBLIME", "TRIDENT", "ZERO-CAL", "BIC", "QUAKER", "TRELOSO" "LIFEBUOY" "PIRACANJUBA" "ACONCHEGO", "DISOFLOR", "PARANA", "HALLS", "ELETR", "KOLENE", "FRINENSE", "PETRIBU", "ATOL", "AZULIM", "NINHO", "STA.CLARA", "LACTA", "TORCIDA", "STELLA DORO", "TIO JOAO", "PALMEIRON", "CHOKOLOT", "ITAIPAVA", "CLOSE UP", "MARAJOARA", "BOAVISTENSE", "PINHO SOL", "FARMAX", "BARUEL", "SCALA", "ALIANCA", "LIFEBUOY", "PROTEX", "SORRISO", "ARIEL", "3M", "BRILHUS", "PITU", "PAJEU", "BRILHANTE", "FLORAL", "LISTERINE", "INTIMUS", "SUAVE", "MOLICO", "MONANGE", "PAIXAO", "GALLO" "JOHNSONS BABY", "GELLUS" "CLUB SOCIAL", "KNORR", "PANSONIC", "VELUD", "FREEG", "FREEGELLS", "MAXIMO BRILHO", "IDEAL", "URCA", "WYDA", "FRUCTUS", "DAFRUTA", "MINHOTO", "ITALAC", "GABOARDI", "MUNDIAL", "PANTENE", "MARILAN", "PIRACANJUBA", "OURO VERDE", "IGUAL", "TODDYNHO", "BAUDUCO", "FACEPA"  # Add more brands as needed
] 
# TODO: The brand "SONHO" is a problem, because there's some stuff that contains it that's not a brand for e.g "SONHO DE VALSA", other problem is this one "QUERO" because "QUERO" is a ordinary word that could be use in different contexts
# TODO: The brand "SUAVE" has a problem alike the one above
# TODO: The brands that are a not apart othe string example ".PROTEX" isn't being get, check and fix that

# Load knows metrics
KNOWN_METRICS = [
    "UN", "CX", "PT", "PCT", "PR", "CXT", "CT", "CXA" , "KG", "G", "M"
]

def changing_db_columns():
    # Database connection
    cur = conn.cursor()

    try: 
        # Fetch products from database 
        # TODO: Instead of getting all products info get only the name of products
        # cur.execute("""SELECT * FROM products""")
        cur.execute("""SELECT * FROM products_with_embeddings_bigger""")
        rows = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Error fetching data from database: {e}")
        rows = []
    # conn.commit()

    # Extract product ids and names
    products_ids = [row[0] for row in rows]
    product_list_simple = [row[1] for row in rows]

    def parse_product_name(product_name):
        """Parses a product name into its components."""
        product_name = product_name.strip()

        brand = None
        metric = None
        quantity = None
        
        # Check for known brands
        for known_brand in KNOWN_BRANDS:
            if known_brand in product_name:
                brand = known_brand
                product_name = product_name.replace(known_brand, '').strip()
                break
        
        # Identify quantity (e.x., "6X30", "1X24 500G")
        quantity_match = re.search(
            r"""
            \b\d+\s?X\s?\d+\s?[A-Za-z]+\b |  # Matches formats like '1 X 24 500G'
            \b\d+\s?X\s?\d+\b(?:\s?\d+\s?[A-Za-z]+)? | # Matches formats like '1 X 24' with optional units (e.g., '500G')
            \b\d+\s?(KG|G|ML|L|MT)\b         # Matches standalone units like '500G'
            """, 
            product_name, 
            re.IGNORECASE | re.VERBOSE
        )
        
        # if quantity_match:
        #     quantity = quantity_match.group()
        #     product_name = product_name.replace(quantity, '').strip()
        
        # # Check for known metrics
        # if quantity:
        #     for known_metric in KNOWN_METRICS:
        #         metric_match = re.search(rf"\b{known_metric}\b", product_name)
                
        #         if known_metric in quantity.upper():
        #             metric = known_metric
        #             break
                
        #         if metric_match:
        #             metric = metric_match.group()
        #             product_name = product_name.replace(metric, '').strip()
        #             break
        # else:
        #     for known_metric in KNOWN_METRICS:
        #         metric_match = re.search(rf"\b{known_metric}\b", product_name)
                
        #         if metric_match:
        #                 metric = metric_match.group()
        #                 product_name = product_name.replace(metric, '').strip()
        #                 break
        
        # Remaining text is considered the name
        name = product_name.strip()
        
        return name, brand or "", metric or "", quantity or ""
    
    # Parse products
    parsed_products = []
    # for product in product_list_simple:
    for product_id, product_name in zip(products_ids, product_list_simple):
        name, brand, metric, quantity = parse_product_name(product_name)
        
        parsed_products.append({
            "id": product_id,
            "name": name,
            "brand": brand,
            "metric": metric,
            "quantity": quantity
        })

    # Print results
    for parsed_product in parsed_products:
        print(f"ID: {parsed_product['id']}, Name: {parsed_product['name']}, Brand: {parsed_product['brand']}, Metric: {parsed_product['metric']}, Quantity: {parsed_product['quantity']}")
        
        sql = """
                UPDATE products_with_embeddings_bigger SET 
                name=%s,
                brand=%s
                WHERE id=%s
        """
        # sql = """
        #         UPDATE products_with_embeddings SET 
        #         name=%s,
        #         brand=%s,
        #         quantity=%s
        #         WHERE product_id=%s
                
        # """
            
        try:
            with conn.cursor() as cur:
                print(f"SQL Query: {sql}")
                cur.execute(sql, (parsed_product['name'], parsed_product['brand'], parsed_product['id']))
                
            # with conn.cursor() as cur:
            #     print(f"SQL Query: {sql}")
            #     cur.execute(sql, (parsed_product['name'], parsed_product['brand'], parsed_product['quantity'], parsed_product['id']))
            # conn.commit()
            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)


# print(changing_db_columns())
