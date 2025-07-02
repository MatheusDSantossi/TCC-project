from rest_framework import mixins, viewsets, generics, permissions
from users.serializers import serializers, InviteDetailSerializer, InviteRegisterSerializer
from users.models import Invite
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from rest_framework.exceptions import ValidationError

from django.utils import timezone

User = get_user_model()

class InviteViewSet(mixins.RetrieveModelMixin, mixins.CreateModelMixin, viewsets.GenericViewSet):
    """
    retrieve: GET / invites/{token}/    -> validate token
    create: POST /invites/register/     -> consume token + create user
    """
    queryset = Invite.objects.filter(used=False)
    lookup_field = 'token'

    def get_serializer_class(self):
        if self.action == "retrieve":
            return InviteDetailSerializer
        
        return InviteRegisterSerializer

    def perform_create(self, serializer):
        """Override CreateModelMixin to handle token logic"""
        token = serializer.validated_data["token"]
        password = serializer.validated_data["password"]
        
        try:
            invite = Invite.objects.get(token=token, used=False)
        except Invite.DoesNotExist:
            raise ValidationError("Invalid or already used invitation.")
        
        if invite.expires_at and invite.expieres_at < timezone.now():
            raise ValidationError("Invitation has expired.")
        
        user = User.objects.create_user(
            email=invite.email,
            password=password,
            business=invite.business,
        )
        invite.used = True
        invite.save(update_fileds=["used"])
    
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

class InviteDetailView(generics.RetrieveAPIView):
    permission_classes = [permissions.AllowAny]
    # queryset = Invite.objects.filter(used=False)
    lookup_field = 'token'
    serializer_class = InviteDetailSerializer
    
    def get_object(self):
        try:
            invite = Invite.objects.get(token=self.kwargs["token"], used=False)
        
        except Invite.DoesNotExist:
            raise NameError("Link de convite inválido ou expirado.")
        
        if invite.expires_at and invite.expires_at < timezone.now():
            raise NameError("Link de convite inválido ou expirado.")
        
        return invite
