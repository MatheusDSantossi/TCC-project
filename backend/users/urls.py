from django.contrib import admin
from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()
router.register('register', RegisterViewSet, basename='register')
router.register('login', LoginViewset, basename='login')
router.register('users', UserViewset, basename='users')


urlpatterns = [
    path('invite/<uuid:token>/', InviteDetailView.as_view(), name='invite-detail'),
    path('register/invite/', InviteRegisterView.as_view(), name='invite-register'),
]

urlpatterns += router.urls
