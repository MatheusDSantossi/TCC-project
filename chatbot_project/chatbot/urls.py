from django.urls import path
from .views.index import index

urlpatterns = [
    path('', index, name='index'),  # Definindo a rota para a view 'index'
]
