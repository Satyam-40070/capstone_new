from django.urls import path

from .views import *

urlpatterns = [
    path("xray/", ImageResizeView.as_view(), name="xray"),
    path("doc/", ImageResizeView.as_view(), name="doc"),
    path("xrayExtract/", ImageResizeView1.as_view(), name="xrayExtract"),
    path("docExtract/", ImageResizeView.as_view(), name="docExtract"),
]