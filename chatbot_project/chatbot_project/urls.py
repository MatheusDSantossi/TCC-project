from django.contrib import admin
from django.urls import path, include
from chatbot.views.index import index

from chatbot.views.product_registration import ProductRegistration
from chatbot.views.menu_options_registration import MenuOptionRegistration 
from chatbot.views.menu_options_edit import MenuOptionEdit 
from chatbot.views.products_view import *
from chatbot.views.menu_options_edit import MenuOptionEdit 
from chatbot.views.employee_registration import EmployeeRegistration, EmployeeLogin, EmployeeDetails 


urlpatterns = [
    path("admin/", admin.site.urls),
    # path("cadastrar_produto/", views.product, name='product'),
    path("", include("chatbot.urls")),
    
    # path("index/", index.as_view(), name='index'),
    path("cadastrar_produto/", ProductRegistration, name='productRegistration'),
    path("products/", ProductListView.as_view(), name='productList'),
    
    path("cadastrar_menu_options/", MenuOptionRegistration.as_view(), name='menuOptionRegistration'),
    
    path("edit_menu_options_list/", MenuOptionEdit.as_view(), name='editMenuOptionsList'),
 
    path('save-menu-order/', MenuOptionEdit.save_menu_order, name='saveMenuOrder'),
    
    path("api/products/", ProductListCreateView.as_view(), name='productListCreate'),
    path("api/products/<int:pk>/", ProductRetrieveUpdateDestroyView.as_view(), name='productRetrieveUpdateDestroy'),
    
    path("register_employee/", EmployeeRegistration.as_view(), name='employeeRegistration'),
    path("login_employee/", EmployeeLogin.as_view(), name='employeeLogin'),
    path("employee_details/", EmployeeDetails.as_view(), name='employeeDetails'),
]
