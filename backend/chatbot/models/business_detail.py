from django.db import models
from core.models import Business

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
