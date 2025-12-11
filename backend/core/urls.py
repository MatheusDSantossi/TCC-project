from django.urls import path, include

from .views import BusinessListCreateView

urlpatterns = [     
    path('business/', BusinessListCreateView.as_view(),
    name='business-list'),
]
