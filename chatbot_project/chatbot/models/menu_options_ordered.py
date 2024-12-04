from django.db import models
from django.shortcuts import render

# HERE I WANT TO CREATE A DB TABLE THAT WILL CONTAIN 
# THE FIVE OPTIONS IN THE CHATBOT, WHAT I MEAN IS IN MY
# CHATBOT TEMPLATE INSTEAD OF MENUOPTION I'LL GET THIS 
# MODEL MENUOPTIONORDERED WHICH WILL BE A TABLE THAT WILL
# BE DROPPED AND CREATED EVERY TIME

class MenuOptionOrdered(models.Model):
    
    menu_ordered_id = models.AutoField(primary_key=True)
    menu_option_description = models.CharField(max_length=255, default='')
    next_options = models.ManyToManyField('self', blank=True, symmetrical=False)

    @staticmethod
    def get_menuOption_by_id(ids):
        return MenuOptionOrdered.objects.filter(id__in=ids)
    
    @staticmethod
    def get_all_menuOptions():
        return MenuOptionOrdered.objects.all()
    
    class Meta:
        db_table = 'menu_options_ordered'
