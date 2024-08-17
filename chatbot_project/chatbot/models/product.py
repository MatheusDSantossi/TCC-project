from django.db import models
from django.contrib.postgres.fields import ArrayField
from project_ecommerce_AI import generate_synonymous, embed_fn, pre_process_prompt_grammatical

class Product(models.Model):
    # productName = models.CharField(max_length=200)
    # price = models.DecimalField(max_digits=10, decimal_places=2)
    # description = models.TextField()
    # image = models.CharField(max_length=5000, null=True, blank=True)
    
    product_id = models.AutoField(primary_key=True)
    nome = models.CharField(max_length=255)
    secao_produto = models.CharField(max_length=255)
    preco_venda = models.DecimalField(max_digits=10, decimal_places=2)
    unidade_medida = models.CharField(max_length=255)
    disponivel = models.BooleanField()
    embeddings = ArrayField(models.FloatField())
    synonymous = ArrayField(models.TextField())

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
    
    @staticmethod
    def generate_synonymous(product):
        generate_synonymous(product)

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
