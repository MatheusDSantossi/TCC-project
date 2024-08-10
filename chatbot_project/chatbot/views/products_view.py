from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View

from django.views import generic
from rest_framework import generics
from ..models.product import *
from .serializers import ProductSerializer

class ProductListView(generic.ListView):
    model = Product
    context_object_name = "product_list"
    
    queryset = Product.objects.all()
    
    template_name = "product_generic_list.html"

class ProductListCreateView(generics.ListCreateAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ProductRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
