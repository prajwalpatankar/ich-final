from django.db import models
from django.db.models.fields import AutoField
from django.forms.fields import FileField


# class MyImage(models.Model):
# 	id = models.AutoField(primary_key=True)
# 	model_pic = models.ImageField(upload_to = '', default='')

class DicomFile(models.Model):
	id = models.AutoField(primary_key=True)
	dicomFile = models.FileField(max_length=None, upload_to='')
