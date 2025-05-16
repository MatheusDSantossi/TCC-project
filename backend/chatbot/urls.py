from django.urls import path, include

from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()
router.register('menu-options', MenuOptionsView, basename='menuOptions')
router.register('register-product', ProductRegistration, basename='registerProduct')
router.register('register-business-detail', BusinessDetailRegistration, basename='registerBusinessDetail')
router.register('chatbot', ChatbotResponse, basename='chatbotResponse')
# urlpatterns = router.urls

urlpatterns = [ 
    path('', include(router.urls)),
    path('update-product/<int:pk>/', ProductUpdate.as_view({'put': 'update_product'}), name='updateProduct'),
    path('delete-product/<int:pk>/', ProductDelete.as_view({'delete': 'delete_product'}), name='deleteProduct'),
    
    path('update-business-detail/<int:pk>/', BusinessDetailUpdate.as_view({'put': 'update_business_detail'}), name='updateBusinessDetail'),
    path('delete-business-detail/<int:pk>/', BusinessDetailDelete.as_view({'delete': 'delete_business_detail'}), name='deleteBusinessDetail'),
    
    path("api/products/", ProductListCreateView.as_view(), name='productListCreate'),
    path("api/business-detail/", BusinessDetailListCreateView.as_view(), name='businessDetailListCreate'),
    path("api/products/<int:pk>/", ProductRetrieveUpdateDestroyView.as_view(), name='productRetrieveUpdateDestroy'),
]

print(f"urlpatters: {urlpatterns}")
