from django.conf.urls import include
from ich_backend import views
from rest_framework import routers
from django.urls import path


router = routers.DefaultRouter()
# router.register(r'upload', views.ImageCreateAPIView)
router.register(r'uploaddicom', views.UploadDicomViewset)


urlpatterns = [
    path('', include(router.urls)),    
    path('model_output', views.model_call),
]
