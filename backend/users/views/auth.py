from rest_framework import viewsets, mixins, permissions
from rest_framework.response import Response
from users.serializers import LoginSerializer, RegisterSerializer
from django.contrib.auth import get_user_model

User = get_user_model()

class LoginViewset(viewsets.ViewSet):
    """
    POST /users/login/  -> validate credentials & return token
    """
    permission_classes = [permissions.AllowAny]
    serializer_class = LoginSerializer
    
    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        # Now the serializer.data is {'email': ..., 'token': ..., 'business': ...}
        
        return Response(serializer.validated_data)
        
class RegisterViewSet(mixins.CreateModelMixin, viewsets.GenericViewSet):
    """
    POST /users/register/   -> create a new user
    """
    queryset = User.objects.all()
    serializer = RegisterSerializer
    permission_classes = [permissions.AllowAny]

            
class UserViewset(viewsets.ViewSet):
    """
    GET /users/     -> list users (admin or appropriate perms)
    GET /users/{pk}/    -> retrieve a single user
    """
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.IsAuthenticated]
