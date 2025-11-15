from django.urls import path, include

from rest_framework.routers import DefaultRouter
from .views import (
    ChatbotResponse,
    MenuOptionsView,
    BusinessDetailListCreateView,
    BusinessDetailRetrieveUpdateDestroyView,
    ProductRetrieveUpdateDestroyView,
    ProductListCreateView,
    ProductImportView,
)

router = DefaultRouter()
router.register("menu-options", MenuOptionsView, basename="menuOptions")
router.register(r"chatbot", ChatbotResponse, basename="chatbot")

urlpatterns = [
    path("", include(router.urls)),
    path(
        "api/business-detail/",
        BusinessDetailListCreateView.as_view(),
        name="businessDetail-list-create",
    ),
    path(
        "api/business-detail/<int:pk>/",
        BusinessDetailRetrieveUpdateDestroyView.as_view(),
        name="businessDetailRetrieveUpdateDestroy",
    ),
    path("api/products/", ProductListCreateView.as_view(), name="productListCreate"),
    path(
        "api/products/<int:pk>/",
        ProductRetrieveUpdateDestroyView.as_view(),
        name="productRetrieveUpdateDestroy",
    ),
    path("api/products/import/", ProductImportView.as_view(), name="productsImport"),
]
