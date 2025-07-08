from rest_framework import generics

from .models import Business
from .serializers import BusinessSerializer

from rest_framework import permissions

class BusinessListCreateView(generics.ListCreateAPIView):
    """
    GET /api/business/ -> list details (allows everyone)
    POST /api/business-detail/ -> create a new business
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
