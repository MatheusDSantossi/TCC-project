from rest_framework import serializers
from .models import Product, BusinessDetail
from django.contrib.auth import get_user_model

User = get_user_model()

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['product_id', 'name', 'brand', 'quantity', 'section', 'price', 'unit', 'stock', 'synonymous', 'embeddings']

class BusinessDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = BusinessDetail
        # fields = '__all__'
        fields = ['id', 'key', 'value', 'business']
        
class ChatbotRequestSerializer(serializers.Serializer):
    user_input = serializers.CharField(max_length=1000)
    text_related_to = serializers.CharField(max_length=255, required=False, allow_null=True)
    business_slug = serializers.SlugField(required=False, allow_null=True)
