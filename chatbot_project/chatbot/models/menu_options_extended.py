from django.db import models
from django.shortcuts import render

# HERE I WANT TO CREATE A DB TABLE THAT WILL CONTAIN 
# THE FOUR OPTIONS IN THE CHATBOT, WHAT I MEAN IS IN MY
# CHATBOT TEMPLATE INSTEAD OF MENUOPTION I'LL GET THIS 
# MODEL MenuOptionExtended WHICH WILL BE A TABLE THAT WILL
# BE DROPPED AND CREATED EVERY TIME

class MenuOptionExtended(models.Model):
    
    menu_extended_id = models.AutoField(primary_key=True, db_column="menu_extended_id")
    menu_option_description = models.CharField(max_length=255, default='')
    next_options = models.ManyToManyField('self', blank=True, symmetrical=False)

    @staticmethod
    def createMenusOptions():
        # First options
        menu_option1_s1 = MenuOptionExtended.objects.create(menu_option_description='Informações sobre o produto')
        # Tudo sobre produto
        menu_option2_s1 = MenuOptionExtended.objects.create(menu_option_description='Verificar disponibilidade de produtos')
        menu_option3_s1 = MenuOptionExtended.objects.create(menu_option_description='Quero saber mais sobre o mercado')
        menu_option4_s1 = MenuOptionExtended.objects.create(menu_option_description='Quero algumas sugestões de perguntas')
        # uma lista de perguntas (sugestões de perguntas com interrogação in the end?) uma lista com tudo que it's able to answer
        
        # Saber mais sobre os produtos
        menu_option1_s2_o1 = MenuOptionExtended.objects.create(menu_option_description='Sobre o nome do produto')
        # digite seu o nome do produto e reforçar essa ação
        menu_option1_s2_o2 = MenuOptionExtended.objects.create(menu_option_description='Sobre a seção do produto')
        menu_option1_s2_o3 = MenuOptionExtended.objects.create(menu_option_description='Sobre a descrição do produto')
        menu_option1_s2_o4 = MenuOptionExtended.objects.create(menu_option_description='Sobre o código do produto')
        ####
        
        # Verificar disponibilidade de produtos
        menu_option2_s2_o1 = MenuOptionExtended.objects.create(menu_option_description='Sobre o nome de um produto especifico')
        menu_option2_s2_o2 = MenuOptionExtended.objects.create(menu_option_description='Quero saber sobre todos os produtos')
        ###
        
        # Quero algumas sugestões de perguntas
        menu_option3_s2_o1 = MenuOptionExtended.objects.create(menu_option_description='Vou escolher a área')
        menu_option3_s2_o2 = MenuOptionExtended.objects.create(menu_option_description='Sobre a pergunta sobre o mercado')
        ###
        
        # Quero saber mais sobre o mercado
        menu_option4_s2_o1 = MenuOptionExtended.objects.create(menu_option_description='Nome do gerente')
        menu_option4_s2_o2 = MenuOptionExtended.objects.create(menu_option_description='Digitar a pergunta sobre o mercado')
        ###
        
        # Saber mais sobre os produtos
        menu_option1_s1.next_options.add(menu_option1_s2_o1, menu_option1_s2_o2, menu_option1_s2_o3, menu_option1_s2_o4)
        ####
        
        # Verificar disponibilidade de produtos
        menu_option2_s1.next_options.add(menu_option2_s2_o1, menu_option2_s2_o2)
        
        # Quero algumas sugestões de perguntas
        menu_option3_s1.next_options.add(menu_option3_s2_o1, menu_option3_s2_o2)
        
        menu_option4_s1.next_options.add(menu_option4_s2_o1, menu_option4_s2_o2)
        

    # menu_option1.next_options.add(menu_option2, menu_option3)

    @staticmethod
    def get_menuOption_by_id(ids):
        return MenuOptionExtended.objects.filter(id__in=ids)
    
    @staticmethod
    def get_all_menuOptions():
        return MenuOptionExtended.objects.all()
    
    class Meta:
        db_table = 'menu_options_extended'
