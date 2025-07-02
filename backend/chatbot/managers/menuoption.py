from django.db import models

class MenuOptionQuerySet(models.QuerySet):
    def by_ids(self, ids):
        return self.filter(pk__in=ids)
    
    def by_description(self, desc):
        return self.get(description__index=desc)
    
class MenuOptionManager(models.Manager):
    def get_queryset(self):
        return MenuOptionQuerySet(self.model, using=self.db)
    
    def get_or_create_description(self, desc: str):
        """Get or create by description, logging which happened"""
        obj, created = self.get_or_create(description=desc)
        action = "Created" if created else "Reused"
        print(f"{action} menu option: {desc}")
        return obj
