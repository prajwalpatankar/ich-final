from rest_framework import serializers
from rest_framework.serializers import (
      ModelSerializer,
)
from .models import *

# class imageSerializer(ModelSerializer):
#    class Meta:
#       model = MyImage
#       fields = [
#          'id',
#          'model_pic'        
#       ]

class dicomSerializer(ModelSerializer):
   class Meta:
      model = DicomFile
      fields = [
         'id',
         'dicomFile'
      ]
