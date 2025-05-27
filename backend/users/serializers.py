from rest_framework import serializers
from .models import *
from knox.models import AuthToken
from django.contrib.auth import get_user_model, authenticate
User = get_user_model()

class InviteDetailSerializer(serializers.ModelSerializer):
    business = serializers.SlugRelatedField(read_only=True, slug_field='slug')
    
    class Meta:
        model = Invite
        fields = ('email', 'business')

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    token = serializers.CharField(read_only=True)
    business = serializers.CharField(read_only=True)
    
    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret.pop('password', None)
        return ret
    
    def validate(self, attrs):
        # Extract raw credentials
        email = attrs.get('email')
        password = attrs.get('password')
        
        user = authenticate(email=email, password=password)
        
        if not user:
            raise serializers.ValidationError("Invalid credentials")
        
        # Generate a token
        _, token = AuthToken.objects.create(user)
        
        return {
            "email": user.email,
            "token": token,
            "business": user.business.slug if user.business else None,
        }

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'email', 'password', 'last_login')
        extra_kwargs = {'password': {'write_only': True}}
        
    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user
        