import re
import psycopg2
from config.database_connection import conn

# Load known brands if available
KNOWN_BRANDS = [ "ANTARCT", "LOPES", "CAMIL", "FUGINI", "BABY ROGER", "OUROLUX", "AMAFIL", "SORA", "YPE", "PILAO", "BEM TE VI", "PURO SABOR", "GAROTO", "BAYGON", "HELLMANNS", "ASSOLAN", "PALACIO", "COQUEIRO", "POPULAR", "VITAMASSA", "TOBOGA", "BRILUX", "OMO", "POLITRIZ", "ARCOVERDE", "BETANIA", "CAMPONESA", "PRETINHO", "NESTLE", "VITARELLA", "PRIMOR", "DELICIA", "FORTALEZA", "EVEN", "LUX", "PALMOLIVE", "PHEBO", "TANG", "ALMA FLORES", "SENADOR", "ROSA BRANCA", "SEDA", "NIVEA", "LEGO", "KIT KAT", "DOLCE GUSTO", "NUTELLA", "COCO LÁ", "COSMOPOLITAN", "L'OREAL", "SABO", "WONDER", "TRES COROAS", "LEITE COOP", "CERELAC", "FRUTIGRA", "LIVIA", "CALDENTE", "SABORINHO", "ARAGOA", "PANAMERICANA", "PENFOLDS", "SABO" "COSIPA", "CRESIL" "BRASIL CHEESE", "WONDER", "CRESIL" "SABO", "PANAMERICANA", "ARAGOA", "COSIPA" "SABO" "CRESIL", "COSIPA", "PANAMERICANA", "ARAGOA" "CRESIL", "COSIPA", "PANAMERICANA", "ARAGOA" "CRESIL", "COSIPA", "PANAMERICANA", "ARAGOA", "REXONA", "SUAVE", "BARUEL" "STA CLARA", "SANTA CLARA", "MARATA", "LACTA", "CAPRICE", "COLGATE", "TUCS", "DOVE", "CLEAR", "SKALA", "NATHALYA", "TIXAN", "PETINHO", "MILI", "PRIVAX", "YOKI", "FRISCO", "FREEGELLS", "EMARE", "NUTRIMEL", "ARCOR", "TRESEMME", "TABU", "P&G", "SOFRUTA", "PILAR", "SBP", "EMBARE", "SUBLIME", "TRIDENT", "ZERO-CAL", "BIC", "QUAKER", "TRELOSO" "LIFEBUOY" "PIRACANJUBA" "ACONCHEGO", "DISOFLOR", "PARANA", "HALLS", "QUERO", "ELETR", "KOLENE", "FRINENSE", "PETRIBU", "ATOL", "AZULIM", "NINHO", "STA.CLARA", "LACTA", "TORCIDA", "STELLA DORO", "TIO JOAO", "PALMEIRON", "CHOKOLOT", "ITAIPAVA", "CLOSE UP", "MARAJOARA", "BOAVISTENSE", "PINHO SOL"  # Add more brands as needed
] 
# TODO: The brand "SONHO" is a problem, because there's some stuff that contains it that's not a brand for e.g "SONHO DE VALSA"

# Load knows metrics
KNOWN_METRICS = [
    "UN", "CX", "PT", "PCT", "PR", "CXT", "CT", "CXA" , "KG", "G", "M"
]

# Database connection
cur = conn.cursor()

# Fetch products from database 
# TODO: Instead of getting all products get only the name of products
cur.execute("""SELECT * FROM products""")
rows = cur.fetchall()
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
    quantity_match = re.search(r"\b\d+X\d+[A-Za-z]*\b|\b\d+\s?(KG|G|ML|L)\b", product_name, re.IGNORECASE)

    
    if quantity_match:
        quantity = quantity_match.group()
        product_name = product_name.replace(quantity, '').strip()
    
    # Check for known metrics
    if quantity:
        for known_metric in KNOWN_METRICS:
            metric_match = re.search(rf"\b{known_metric}\b", product_name)
            
            if known_metric in quantity.upper():
                metric = known_metric
                break
            
            if metric_match:
                metric = metric_match.group()
                product_name = product_name.replace(metric, '').strip()
                break
    else:
        for known_metric in KNOWN_METRICS:
            metric_match = re.search(rf"\b{known_metric}\b", product_name)
            
            if metric_match:
                    metric = metric_match.group()
                    product_name = product_name.replace(metric, '').strip()
                    break
    
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
            UPDATE products_with_embeddings SET 
            name=%s,
            brand=%s,
            quantity=%s
            WHERE product_id=%s
            
    """
        
    try:
        with conn.cursor() as cur:
            print(f"SQL Query: {sql}")
            cur.execute(sql, (parsed_product['name'], parsed_product['brand'], parsed_product['quantity'], parsed_product['id']))
        # conn.commit()
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


