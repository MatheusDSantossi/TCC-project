from django.urls import path, include

from .views import BusinessListCreateView

urlpatterns = [     
    path('api/business/', BusinessListCreateView.as_view(),
    name='business-list'),
]
