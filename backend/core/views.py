from rest_framework import generics

from .models import Business
from .serializers import BusinessSerializer

from rest_framework import permissions

class BusinessListView(generics.ListAPIView):
    """
    GET /api/business/ -> list businesses (public: the store picker in the frontend)

    Read-only on purpose: creating a business is an admin task (Django shell),
    not something an anonymous visitor can do.
    """
    permission_classes = [permissions.AllowAny]
    queryset = Business.objects.all()
    serializer_class = BusinessSerializer
    ''
        
class BusinessDetailRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    """
    GET    /api/business/<pk>/    → retrieve one detail (must belong to request.user.business)
    PUT    /api/business/<pk>/    → update it
    DELETE /api/business/<pk>/    → delete it
    """
    permission_classes = [permissions.IsAuthenticated]
    queryset = Business.objects.all()
    serializer_class = BusinessSerializer
