from django.db import models
from django.shortcuts import render
from django.contrib.postgres.fields import ArrayField
from project_ecommerce_AI import generate_synonymous, embed_fn, pre_process_prompt_grammatical, fetch_products_alike

from ..models import search_image

class Product(models.Model):
    # productName = models.CharField(max_length=200)
    # price = models.DecimalField(max_digits=10, decimal_places=2)
    # description = models.TextField()
    # image = models.CharField(max_length=5000, null=True, blank=True)
    
    product_id = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=255, default='')
    secao_produto = models.CharField(max_length=255, default='')
    preco_venda = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    unidade_medida = models.CharField(max_length=255, default='')
    disponivel = models.BooleanField(default=True)
    embeddings = ArrayField(models.FloatField(), default=list)
    synonymous = ArrayField(models.TextField(), default=list)

    @staticmethod
    def get_product_by_id(ids):
        return Product.objects.filter(id__in=ids)

    
    @staticmethod
    def get_all_products():
        return Product.objects.all()
    
    @staticmethod
    def get_product_by_name(name):
        if name:
            return Product.objects.filter(nome=name)
        return Product.get_all_products()
    
    def get_product_by_section(section):
        if section:
            return Product.objects.filter(secao_produto=section)
        return None
    
    @staticmethod
    def generate_synonymous(product):
        generate_synonymous(product)

    @staticmethod
    def fetch_products_alike(product_name):
        print(fetch_products_alike(product_name))
        
    @staticmethod
    def get_synonymus(name):
        product = Product.objects.filter(nome=name).first()
        if product:
            return product.synonymous
        return None
    
    @staticmethod
    def generate_emebed(productName):
        # embed_fn(product.nome)
        return embed_fn(productName)
    
    @staticmethod
    def fix_product_name(productName):
        return pre_process_prompt_grammatical(productName)
    
    
    class Meta:
        db_table = 'products_with_embeddings'

def product_carousel_view(request):
    products = Product.objects.all()
    
    for product in products:
        if not product.image_url: 
            product.image_url = search_image(product.productName)
            product.save()
            
    context = {'products': products}
    return render(request, 'ecomerce/e-commerece_template.html', context)
