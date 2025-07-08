# from pathlib import Path
# import sys
# import os

# # Build paths inside the project like this: BASE_DIR / 'subdir'.
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Ensure the root directory is included
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import logging
from config.database_connection import conn

from django.core.management.base import BaseCommand
from django.db import transaction

from chatbot.models import Product
from backend.utils.brands_utils import find_brands_in_text
from backend.utils.metrics_utils import extract_metric_and_quantity

logger = logging.getLogger(__name__)

def parse_product_name(raw_name: str) -> dict:
        """
        Decompose raw_name into its components.
        Returns dict with keys: product_id*, name, brand, quantity
        """
        name = raw_name.strip()
        
        # 1) Extract brand (take the first match in the text)
        brands = find_brands_in_text(name)
        brand = brands[0] if brands else ""
        
        if brand:
            # remove only the first occurrence
            name = re.sub(rf"\b{re.escape(brand)}\b", "", name, count=1, flags=re.IGNORECASE).strip()
            
        # 2) Extract quantity
        name, quantity = extract_metric_and_quantity(name)
        
        return {"name": name, "brand": brand, "quantity": quantity}            

class Command(BaseCommand):
    help = "Parse product names into name, brand, quantity and update the DB."
    
    def add_arguments(self, parser):
        parser.add_argument("--business-id", type=int, required=True, help="Only process products for this business ID.")
        
    def handle(self, *args, **options):
        business_id = options["business_id"]
        qs = Product.objects.filter(business_id=business_id)
        
        self.stdout.write(f"Found {qs.count()} products to process.")
        
        for product in qs.iterator():
            parsed = parse_product_name(product.name)
            
            # Only update if something changed
            if (
                parsed["name"] != product.name or parsed["brand"] != product.brand or parsed["quantity"] != product.quantity or parsed["metric"] != product.metric
            ):
                self.update_product(product, parsed)
                
            self.stdout.write(self.style.SUCCESS("Done"))

    @transaction.atomic
    def update_product(self, product, parsed):
        logger.info(f"Updating product {product.product_id}: {parsed}")
        
        product.name = parsed["name"]
        product.brand = parsed["brand"]
        product.quantity = parsed["quantity"]

        product.save(update_fields=["name", "brand", "quantity"])

# def changing_db_columns():
#     # Database connection
#     cur = conn.cursor()

#     try: 
#         # Fetch products ID and NAME from database 
#         cur.execute("""SELECT product_id, name FROM products_with_embeddings WHERE business_id = 2""")
#         rows = cur.fetchall()
#     except psycopg2.Error as e:
#         print(f"Error fetching data from database: {e}")
#         rows = []
#     # conn.commit()

#     # Extract product ids and names
#     products_ids = [row[0] for row in rows]
#     product_list_simple = [row[1] for row in rows]

#     def parse_product_name(product_name):
#         """Parses a product name into its components."""
#         product_name = product_name.strip()

#         brand = None
#         metric = None
#         quantity = None
        
#         # Check for known brands
#         for known_brand in KNOWN_BRANDS:
#             if known_brand in product_name:
#                 brand = known_brand
#                 product_name = product_name.replace(known_brand, '').strip()
#                 break
        
#         # Identify quantity (e.x., "6X30", "1X24 500G")
#         quantity_match = re.search(
#             r"""
#             \b\d+\s?X\s?\d+\s?[A-Za-z]+\b |  # Matches formats like '1 X 24 500G'
#             \b\d+\s?X\s?\d+\b(?:\s?\d+\s?[A-Za-z]+)? | # Matches formats like '1 X 24' with optional units (e.g., '500G')
#             \b\d+\s?(KG|G|ML|L|MT)\b         # Matches standalone units like '500G'
#             """, 
#             product_name, 
#             re.IGNORECASE | re.VERBOSE
#         )
        
#         if quantity_match:
#             quantity = quantity_match.group()
#             product_name = product_name.replace(quantity, '').strip()
        
#         # Check for known metrics
#         if quantity:
#             for known_metric in KNOWN_METRICS:
#                 metric_match = re.search(rf"\b{known_metric}\b", product_name)
                
#                 if known_metric in quantity.upper():
#                     metric = known_metric
#                     break
                
#                 if metric_match:
#                     metric = metric_match.group()
#                     product_name = product_name.replace(metric, '').strip()
#                     break
#         else:
#             for known_metric in KNOWN_METRICS:
#                 metric_match = re.search(rf"\b{known_metric}\b", product_name)
                
#                 if metric_match:
#                         metric = metric_match.group()
#                         product_name = product_name.replace(metric, '').strip()
#                         break
        
#         # Remaining text is considered the name
#         name = product_name.strip()
        
#         return name, brand or "", metric or "", quantity or ""
    
#     # Parse products
#     parsed_products = []
#     # for product in product_list_simple:
#     for product_id, product_name in zip(products_ids, product_list_simple):
#         name, brand, metric, quantity = parse_product_name(product_name)
        
#         parsed_products.append({
#             "id": product_id,
#             "name": name,
#             "brand": brand,
#             "metric": metric,
#             "quantity": quantity
#         })

#     # Print results
#     for parsed_product in parsed_products:
#         print(f"ID: {parsed_product['id']}, Name: {parsed_product['name']}, Brand: {parsed_product['brand']}, Metric: {parsed_product['metric']}, Quantity: {parsed_product['quantity']}")
        
#         sql = """
#                 UPDATE products_with_embeddings SET 
#                 name=%s,
#                 brand=%s,
#                 quantity=%s
#                 WHERE product_id=%s AND business_id = 2
                
#         """
            
#         try:
                
#             with conn.cursor() as cur:
#                 print(f"SQL Query: {sql}")
#                 cur.execute(sql, (parsed_product['name'], parsed_product['brand'], parsed_product['quantity'], parsed_product['id']))
                
#             conn.commit()
            
#         except (Exception, psycopg2.DatabaseError) as error:
#             print(error)


# print(changing_db_columns())
