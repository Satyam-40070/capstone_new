from django.urls import path

from .views import *

urlpatterns = [
    path("resize/", ImageResizeView.as_view(), name="resize"),
]