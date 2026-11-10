from django.urls import path, include

from .views import BusinessListView

urlpatterns = [     
    path('business/', BusinessListView.as_view(),
    name='business-list'),
]
