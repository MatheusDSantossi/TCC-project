from django.db import models

class Business(models.Model):
    """
    Represents a tenant / company whose products we serve.
    """
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True, help_text="Unique identifier for the business, Use in URLs or API to identify the business.")
    # Any other metadata (e.g. logo, theme, contact info)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
        