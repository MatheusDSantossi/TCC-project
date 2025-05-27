from django.contrib import admin
from .models import *
from django.core.mail import send_mail
from django.urls import reverse
from django.conf import settings

admin.site.register(CustomUser)

@admin.register(Invite)
class InviteAdmin(admin.ModelAdmin):
    list_display = ('email', 'business', 'used', 'created_at', 'expires_at')
    # search_fields = ('email', 'business__name', 'token')
    # list_filter = ('business', 'created_at')
    # ordering = ('-created_at',)
    
    actions = ['send_invites']
    
    
    def send_invites(self, request, queryset):
        # Frontend URL for the invite registration
        FRONTEND = "http://localhost:5173/"
        
        for invite in queryset:
            link = f"{FRONTEND}register/invite/{invite.token}/"
            # link = request.build_absolute_uri(reverse('invite-detail', args=[invite.token]))
            send_mail(
                subject=f"Your invitation to {invite.business.name}",
                message=f"Please register here: {link}", 
                from_email=None,
                recipient_list=[invite.email],
                )
        self.message_user(request, "Invites sent successfully.")
    send_invites.short_description = "Send invitation emails"
    