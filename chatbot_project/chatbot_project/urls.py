"""
URL configuration for chatbot_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from chatbot.views.index import index
from chatbot.views.product_registration import ProductRegistration
from chatbot.views.products_view import *

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("cadastrar_produto/", views.product, name='product'),
    path("", include("chatbot.urls")),
    
    # path("index/", index.as_view(), name='index'),
    path("cadastrar_produto/", ProductRegistration.as_view(), name='productRegistration'),
    path("products/", ProductListView.as_view(), name='productList'),
    
    path("api/products/", ProductListCreateView.as_view(), name='productListCreate'),
    path("api/products/<int:pk>/", ProductRetrieveUpdateDestroyView.as_view(), name='productRetrieveUpdateDestroy'),
]
