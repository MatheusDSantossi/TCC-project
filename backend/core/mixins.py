class BusinessQuerysetMixin:
    """
    Override get_queryset to filter by request.user.business.
    """

    def get_queryset(self):
        user = self.request.user
        # biz = getattr(self.request.user.business, 'Business', None)
        biz = self.request.user.business
        print(f"BusinessQuerysetMixin: user={user.email!r}, business={biz.slug if biz else None!r}")
        
        qs = super().get_queryset()
        if biz is None:
            return qs.none()
        
        qs = qs.filter(business=biz)
        return qs
        