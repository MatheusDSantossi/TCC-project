from django.urls import path, include

from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()
router.register('menu-options', MenuOptionsView, basename='menuOptions')
router.register('register-product', ProductRegistration, basename='registerProduct')
router.register('chatbot', ChatbotResponse, basename='chatbotResponse')
# urlpatterns = router.urls

urlpatterns = [ 
    path('', include(router.urls)),
    path('update-product/<int:pk>/', ProductUpdate.as_view({'put': 'update_product'}), name='updateProduct'),
    path("api/products/", ProductListCreateView.as_view(), name='productListCreate'),
    path("api/products/<int:pk>/", ProductRetrieveUpdateDestroyView.as_view(), name='productRetrieveUpdateDestroy'),
]

print(f"urlpatters: {urlpatterns}")
