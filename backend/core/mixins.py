class BusinessQuerysetMixin:
    """
    Override get_queryset to filter by request.user.business.
    """

    def get_queryset(self):
        qs = super().get_queryset()
        biz = getattr(self.request.user.business, 'business', None)
        if biz is None:
            return qs.none()
        
        qs = qs.filter(business=biz)
        return qs
        