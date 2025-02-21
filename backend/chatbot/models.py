from django.db import models
from django.shortcuts import render
from django.contrib.postgres.fields import ArrayField
from utils.project_ecommerce_AI import * 

class Product(models.Model):
    
    product_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, default='')
    brand = models.CharField(max_length=255, default='')
    quantity = models.CharField(max_length=50, default='')
    secao_produto = models.CharField(max_length=255, default='')
    preco_venda = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    unidade_medida = models.CharField(max_length=255, default='')
    # disponivel = models.BooleanField(default=True)
    disponivel = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    embeddings = ArrayField(models.FloatField(), default=list)
    synonymous = ArrayField(models.TextField(), default=list, blank=True)

    @staticmethod
    def get_product_by_id(ids):
        return Product.objects.filter(id__in=ids)

    
    @staticmethod
    def get_all_products():
        return Product.objects.all()
    
    @staticmethod
    def get_product_by_name(product_name):
        if product_name:
            # return Product.objects.filter(name=name)
            return Product.objects.get(name=product_name)
        # return Product.get_all_products()
        return None
    
    def get_product_by_section(section):
        if section:
            return Product.objects.filter(secao_produto=section)
        return None
    
    @staticmethod
    def generate_synonymous(product=None, product_list=None, brand=None, synonyms=None):
        if product_list:
            return generate_synonymous(product_list=product_list, brand=brand, synonyms=synonyms)
        else:
            return generate_synonymous(product=product, brand=brand, synonyms=synonyms)

    @staticmethod
    def fetch_products_alike(product_name):
        print(fetch_products_alike(product_name))
        
    @staticmethod
    def get_synonymus(name):
        product = Product.objects.filter(name=name).first()
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


    def __str__(self):
        return super().__str__()

# HERE I WANT TO CREATE A DB TABLE THAT WILL CONTAIN 
# THE FOUR OPTIONS IN THE CHATBOT, WHAT I MEAN IS IN MY
# CHATBOT TEMPLATE INSTEAD OF MENUOPTION I'LL GET THIS 
# MODEL MenuOptionExtended WHICH WILL BE A TABLE THAT WILL
# BE DROPPED AND CREATED EVERY TIME

class MenuOptionExtended(models.Model):
    
    menu_extended_id = models.AutoField(primary_key=True, db_column="menu_extended_id")
    # menu_option_description = models.CharField(max_length=255, default='')
    menu_option_description = models.CharField(max_length=255, default='', unique=True)
    next_options = models.ManyToManyField('self', blank=True, symmetrical=False)

    @staticmethod
    def create_or_get_option(menu_option_description):
        """ Retrieve or create an option based on its description"""
        obj, create = MenuOptionExtended.objects.get_or_create(menu_option_description=menu_option_description)
        if(create):
            print(f"Created new menu option: {menu_option_description}")
        else:
            print(f"Reused existing menu option: {menu_option_description}")
        return obj

    @staticmethod
    def createMenusOptions():
        # Level 1: Main options
        menu_option1_s1 = MenuOptionExtended.create_or_get_option(menu_option_description='Tenho dúvida de um produto específico')
        menu_option2_s1 = MenuOptionExtended.create_or_get_option(menu_option_description=f'Quero informações do {business_name}')
        menu_option3_s1 = MenuOptionExtended.create_or_get_option(menu_option_description=f'Quero saber mais sobre os serviços do {business_name}')
        menu_option4_s1 = MenuOptionExtended.create_or_get_option(menu_option_description=f'Quais os produtos o {business_name} vende?')
        menu_option5_s1 = MenuOptionExtended.create_or_get_option(menu_option_description='Desejo outra opção')
        # uma lista de perguntas (sugestões de perguntas com interrogação in the end?) uma lista com tudo que it's able to answer
        
        # Level 2: Opções específicas para cada categoria
        # Opções para "Quero saber mais sobre os serviços do {business}"
        menu_option1_s2_o1 = MenuOptionExtended.create_or_get_option(menu_option_description='Digitar o nome do produto')
        menu_option1_s2_o2 = MenuOptionExtended.create_or_get_option(menu_option_description='Digitar a seção do produto')
        menu_option1_s2_o3 = MenuOptionExtended.create_or_get_option(menu_option_description='Digitar a marca do produto')
        menu_option1_s2_o4 = MenuOptionExtended.create_or_get_option(menu_option_description='Digitar alguma descrição do produto/sinônimo')
        ####
        
        # Opções para "Quais os produtos vocês vendem?"
        menu_option2_s2_o1 = MenuOptionExtended.create_or_get_option(menu_option_description='Nome do gerente')
        menu_option2_s2_o2 = MenuOptionExtended.create_or_get_option(menu_option_description='Digitar a pergunta sobre o mercado')
        ###
        
        # Quero algumas sugestões de perguntas
        menu_option3_s2_o1 = MenuOptionExtended.create_or_get_option(menu_option_description='Quais são as promoções do dia?')
        menu_option3_s2_o2 = MenuOptionExtended.create_or_get_option(menu_option_description='Quais são as promoções do dia?')
        ###
        
        # Quero saber mais sobre o mercado
        menu_option4_s2_o1 = MenuOptionExtended.create_or_get_option(menu_option_description='Lista de categorias disponíveis')
        menu_option4_s2_o2 = MenuOptionExtended.create_or_get_option(menu_option_description='Quero saber sobre um produto específico')
        ###
        
        ### 
        menu_option5_s2_o1 = MenuOptionExtended.create_or_get_option(menu_option_description='Informar o assunto sobre o qual quero tratar')
        
        # Saber mais sobre os produtos
        menu_option1_s1.next_options.clear()
        menu_option1_s1.next_options.add(menu_option1_s2_o1, menu_option1_s2_o2, menu_option1_s2_o3, menu_option1_s2_o4)
        ####
        
        # Verificar disponibilidade de produtos
        menu_option2_s1.next_options.clear()
        menu_option2_s1.next_options.add(menu_option2_s2_o1, menu_option2_s2_o2)
        
        # Quero algumas sugestões de perguntas
        menu_option3_s1.next_options.clear()
        menu_option3_s1.next_options.add(menu_option3_s2_o1, menu_option3_s2_o2)
        
        menu_option4_s1.next_options.clear()
        menu_option4_s1.next_options.add(menu_option4_s2_o1, menu_option4_s2_o2)
        
        # Me fale sobre qual assunto você quer tratar
        menu_option5_s1.next_options.clear()
        menu_option5_s1.next_options.add(menu_option5_s2_o1)
        
    @staticmethod
    def get_menuOption_by_id(ids):
        return MenuOptionExtended.objects.filter(id__in=ids)
    
    @staticmethod
    def get_all_menuOptions():
        return MenuOptionExtended.objects.all()
    
    def delete_and_recreate_menuOption():
        MenuOptionExtended.objects.all().delete()
        MenuOptionExtended.createMenusOptions()
    
    class Meta:
        db_table = 'menu_options_extended'

# HERE I WANT TO CREATE A DB TABLE THAT WILL CONTAIN 
# THE FIVE OPTIONS IN THE CHATBOT, WHAT I MEAN IS IN MY
# CHATBOT TEMPLATE INSTEAD OF MENUOPTION I'LL GET THIS 
# MODEL MENUOPTIONORDERED WHICH WILL BE A TABLE THAT WILL
# BE DROPPED AND CREATED EVERY TIME

class MenuOptionOrdered(models.Model):
    
    menu_ordered_id = models.AutoField(primary_key=True)
    menu_option_description = models.CharField(max_length=255, default='')

    @staticmethod
    def get_menuOption_by_id(ids):
        return MenuOptionOrdered.objects.filter(id__in=ids)
    
    @staticmethod
    def get_all_menuOptions():
        return MenuOptionOrdered.objects.all()
    
    class Meta:
        db_table = 'menu_options_ordered'
        
