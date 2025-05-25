from rest_framework import permissions

class IsBusinessMember(permissions.BasePermission):
    """
    Custom permission to only allow business members to access the view.
    """

    def has_object_permission(self, request, view, obj):
        return getattr(request.user, 'business', None) == obj.business
