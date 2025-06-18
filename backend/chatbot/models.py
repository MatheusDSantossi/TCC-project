from django.db import models
from django.db.models import JSONField
from utils.project_ecommerce_AI import *
from core.models import Business
from .managers.product import ProductManager

# Products
class Product(models.Model):
    
    product_id = models.AutoField(primary_key=True)
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='%(class)ss',
                                 null=False, help_text="Business to which the product belongs.")

    # Common fields
    name = models.CharField(max_length=255)
    brand = models.CharField(max_length=255, default='')
    quantity = models.CharField(max_length=50, default='')
    section = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    unit = models.CharField(max_length=255, default='')
    stock = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    embeddings = JSONField(models.FloatField(), default=list)
    synonymous = JSONField(models.TextField(), default=list, blank=True, null=True)

    objects = ProductManager()
    # Example of how to use it:
    # Product.objects.by_section("Beverages").available()
    
    class Meta:
        db_table = 'products_with_embeddings'
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.brand})"

# Menu Option
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
        menu_option1_s2_o4 = MenuOptionExtended.create_or_get_option(menu_option_description='Digitar alguma descrição do produto ou sinônimo')
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

# Business 

class BusinessDetail(models.Model):
    
    key = models.CharField(max_length=30, default='')
    value = models.TextField()
    business  = models.ForeignKey(
        Business,
        on_delete=models.CASCADE,
        related_name='details',
        null=True,                # temporarily allow null for existing rows
        help_text="Which business this detail belongs to"
    )

    @staticmethod
    def get_business_detail_by_id(ids):
        return BusinessDetail.objects.filter(id__in=ids)
    
    @staticmethod
    def get_all_business_detail():
        return BusinessDetail.objects.all()
