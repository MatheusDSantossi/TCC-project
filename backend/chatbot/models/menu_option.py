from django.db import models
from managers.menuoption import MenuOptionManager

class MenuOptionExtended(models.Model):
    
    menu_extended_id = models.AutoField(primary_key=True, db_column="menu_extended_id")
    description = models.CharField(max_length=255, default='', unique=True)
    next_options = models.ManyToManyField('self', blank=True, symmetrical=False)
    
    objects = MenuOptionManager()
    
    class Meta:
        db_table = 'menu_options_extended'
        ordering = ['description']
        
    def __str__(self):
        return self.description

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
