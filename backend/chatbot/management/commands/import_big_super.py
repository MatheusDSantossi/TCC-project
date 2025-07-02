# IMPORTANT: Using pandas in this case might be a little
# bad for memory beyond tens of thousands of rows. So, study to use
# openpyxl for streaming
# Example:
# from openpyxl import load_workbook

# wb = load_workbook(opts['csvfile'], read_only=True)
# ws = wb.active  # or wb[sheet_name]

# assume first row is headers:
# headers = [cell.value for cell in next(ws.rows)]
# for row in ws.rows:
#     data = {headers[i]: cell.value for i, cell in enumerate(row)}
#     emb = embed_fn(data['name'])
#     batch.append(BigSuperProduct(
#         business   = biz,
#         name       = data['name'],
#         # … rest of your mapping …
#         embeddings = emb,
#     ))
import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from core.models import Business
from chatbot.models import Product
from utils.embeddings import embed_fn
from utils.project_ecommerce_AI import genai, conn, find_section


# IMPORTANT: Run in python shell:
# BigSuperProduct.objects.filter(business__isnull=True).update(business=biz2)

class Command(BaseCommand):
    help = "Import products from xls into Product, computing embeddings."
    
    def add_arguments(self, parser):
        parser.add_argument('xlsfile', type=str, help="Path to source XLS")
        parser.add_argument('--business', type=str, required=True, help="Business slug for these products")
        parser.add_argument('--batch-size', type=int, default=500, help="How many records to bulk-create per batch")

    def handle(self, *args, **opts):
        biz = Business.objects.get(slug=opts['business'])
        batch, size = [], opts['batch_size']
        
        df = pd.read_excel(opts['xlsfile'], sheet_name=0, dtype=str)
        
        # with open(opts['csvfile'], newline='', encoding='utf-8') as fp:
            # reader = csv.DictReader(fp)
            # for row in reader:
        for _, row in df.iterrows():
            # Compute embedding only once
            emb = embed_fn(find_section, genai, conn, row['nome'])
            
            # batch.append(BigSuperProduct(
            batch.append(Product(
                business      = biz,
                name          = row['nome'],
                # code          = row['codigo'],
                # barcode       = row.get('codigo_barras', ''),
                unit       = row.get('unidade', ''),
                # quantity       = row.get('unidade', ''),
                # total_cost    = row['custo_total'],
                # venda_total   = row['venda_total'],
                # margem_venda  = row['margem_venda'],
                # markup_venda  = row['markup_venda'],
                section = row['secao_produto'],
                preco_venda   = row['preco_venda'],
                # ncm           = row.get('NCM',''),
                # cest          = row.get('CEST',''),
                # preco_custo   = row['preco_custo'],
                # preco_minimo  = row.get('preco_minimo'),
                stock    = row['estoque'],
                # ativo         = row['ativo'] in ('True','true','1'),
                # cadastrado_em = row['cadastrado_em'],
                embeddings    = emb,
            ))
                
        # Flush a batch
        if len(batch) >= size:
            Product.objects.bulk_create(batch)
            self.stdout.write(f"Inserted {len(batch)} records")
            batch = []
                    