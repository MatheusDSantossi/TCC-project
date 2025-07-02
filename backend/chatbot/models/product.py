from django.db import models
from django.db.models import JSONField
from core.models import Business
from chatbot.managers.product import ProductManager

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
