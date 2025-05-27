from django.shortcuts import render
from rest_framework import viewsets, permissions, generics
from .serializers import *
from .models import *
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate
from knox.models import AuthToken
from rest_framework.exceptions import ValidationError

from datetime import timezone

User = get_user_model()

# Invite Views
class InviteDetailView(generics.RetrieveAPIView):
    queryset = Invite.objects.filter(used=False)
    lookup_field = 'token'
    serializer_class = InviteDetailSerializer
    
class InviteRegisterView(generics.CreateAPIView):
    """
    Accepts { token, password } -> creates the user under invite.business
    """
    serializer_class = serializers.Serializer # We'll do it manually
    
    def create(self, request, *args, **kwargs):
        token = request.data.get('token')
        password = request.data.get('password')
        
        try:
            invite = Invite.objects.get(token=token, used=False)
        except Invite.DoesNotExist:
            raise ValidationError("Invalid or used invitation.")
        # Optionally check expiry, for now I'm not going to implement it
        if invite.expires_at and invite.expires_at < timezone.now():
            raise ValidationError("This invitation has expired.")
        
        # Create the user
        user = User.objects.create_user(
            email=invite.email,
            password=password,
            business=invite.business
        )
        
        # Mark invite as used
        invite.used = True
        invite.save(update_fields=['used'])
        
        return Response({"detail": "Registration succesful."}, status=201)

# User Views
class LoginViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = LoginSerializer
    
    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        # Now teh serializer.data is {'email': ..., 'token': ..., 'business': ...}
        
        return Response(serializer.data)
        
        # if serializer.is_valid():
        #     email = serializer.validated_data['email']
        #     password = serializer.validated_data['password']
            
        #     user = authenticate(request, email=email, password=password)
            
        #     if user:
        #         _, token = AuthToken.objects.create(user)
        #         return Response({
        #             'user': self.serializer_class(user).data,
        #             'token': token
        #         })
        #     else:
        #         return Response({'error': 'Invalid credentials'}, status=403)
        # else:
        #     return Response(serializer.errors, status=400)
        
        
class RegisterViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    queryset = User.objects.all()
    serializer = RegisterSerializer
    
    def create(self, request):
        serializer = self.serializer(data=request.data)
        # serializer.is_valid(raise_exception=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            return Response(serializer.errors, status=400)
            
class UserViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]
    # permission_classes = [permissions.AllowAny]
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    
    def list(self, request):
        queryset = User.objects.all()
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)
