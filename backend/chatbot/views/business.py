from rest_framework import generics

from chatbot.models import BusinessDetail
from chatbot.serializers import BusinessDetailSerializer

from rest_framework import permissions
from core.mixins import BusinessQuerysetMixin

class BusinessDetailListCreateView(BusinessQuerysetMixin, generics.ListCreateAPIView):
    """
    GET /api/business-detail/ -> list details for request.user.business
    POST /api/business-detail/ -> create a new detail under request.user.business
    """
    # permission_classes = [permissions.IsAuthenticated]
    permission_classes = [permissions.IsAuthenticated]
    queryset = BusinessDetail.objects.all()
    serializer_class = BusinessDetailSerializer
    
    def perform_create(self, serializer):
        # Automatically set business = request.user.business
        serializer.save(business=self.request.user.business)
        
class BusinessDetailRetrieveUpdateDestroyView(BusinessQuerysetMixin, generics.RetrieveUpdateDestroyAPIView):
    """
    GET    /api/business-detail/<pk>/    → retrieve one detail (must belong to request.user.business)
    PUT    /api/business-detail/<pk>/    → update it
    DELETE /api/business-detail/<pk>/    → delete it
    """
    permission_classes = [permissions.IsAuthenticated]
    queryset = BusinessDetail.objects.all()
    serializer_class = BusinessDetailSerializer
