from django.urls import path
from .views.index import index
from .views.sample_data import sample_data

urlpatterns = [
    path('', index, name='index'),  # Definindo a rota para a view 'index'
    path('api/sample/', sample_data, name='sample_data'),
    # path('ap/menu-options', index, name='menu_options')
]
