from django.shortcuts import get_object_or_404
from chatbot.models import BusinessDetail

def resolve_business(slug: str) -> BusinessDetail:
    """
    Given a slug, return the Business or Raise Http404.
    If slug is None, it can be return a default latter or None.
    """
    if slug:
        return get_object_or_404(BusinessDetail, business=slug)
    return None
