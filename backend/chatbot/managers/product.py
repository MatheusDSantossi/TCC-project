from django.db import models

class ProductQuerySet(models.QuerySet):
    def by_ids(self, ids):
        return self.filter(pk__in=ids)
    def by_name(self, name):
        return self.filter(name__iexact=name)
    def by_section(self, section):
        return self.filter(section__iexact = section)
    def available(self):
        return self.filter(stock__gt=0)
    
class ProductManager(models.Manager):
    def get_queryset(self):
        return ProductQuerySet(self.model, using=self._db)
    
    def by_ids(self, ids):
        return self.get_queryset().by_ids(ids)
    def by_name(self, name):
        return self.get_queryset(name)
    def by_section(self, section):
        return self.get_queryset( section)
    def available(self):
        return self.get_queryset(0)
    