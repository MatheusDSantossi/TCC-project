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
        "business-detail/",
        BusinessDetailListCreateView.as_view(),
        name="businessDetail-list-create",
    ),
    path(
        "business-detail/<int:pk>/",
        BusinessDetailRetrieveUpdateDestroyView.as_view(),
        name="businessDetailRetrieveUpdateDestroy",
    ),
    path("products/", ProductListCreateView.as_view(), name="productListCreate"),
    path(
        "products/<int:pk>/",
        ProductRetrieveUpdateDestroyView.as_view(),
        name="productRetrieveUpdateDestroy",
    ),
    path("products/import/", ProductImportView.as_view(), name="productsImport"),
]
