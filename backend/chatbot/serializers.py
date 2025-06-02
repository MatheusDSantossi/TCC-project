from rest_framework import serializers
from .models import *
from django.contrib.auth import get_user_model

User = get_user_model()

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['product_id', 'name', 'brand', 'quantity', 'secao_produto', 'preco_venda', 'unidade_medida', 'disponivel', 'synonymous']
    
    def create(self, validated_data):
        biz = getattr(self.context['request'].user, 'business', None)
        product = Product.objects.create(business=biz, **validated_data)
        return product

class BusinessDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = BusinessDetail
        # fields = '__all__'
        fields = ['id', 'key', 'value']
