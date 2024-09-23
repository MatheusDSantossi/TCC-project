from django.db import models
from django.shortcuts import render
from project_ecommerce_AI import create_menu_options_prompts

class MenuOption(models.Model):
    
    menu_id = models.AutoField(primary_key=True)
    menu_option_name = models.CharField(max_length=255, default='')
    menu_option_description = models.CharField(max_length=255, default='')

    @staticmethod
    def get_menuOption_by_id(ids):
        return MenuOption.objects.filter(id__in=ids)
    
    @staticmethod
    def get_all_menuOptions():
        return MenuOption.objects.all()
    
    @staticmethod
    def get_menuOption_by_name(name):
        if name:
            return MenuOption.objects.filter(nome=name)
        return MenuOption.get_all_menuOptions()
    
    @staticmethod
    def create_menu_options_prompts():
        return create_menu_options_prompts()
    
    class Meta:
        db_table = 'menu_options'
