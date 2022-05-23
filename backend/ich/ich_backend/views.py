from .serializers import *
from .models import *
from rest_framework import viewsets
# import requests
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer


# import torch 
# import torchvision.transforms as T
# import torchvision.models as models
# from torchvision.utils import make_grid
# from PIL import Image

# class ImageCreateAPIView(viewsets.ModelViewSet):
# 	serializer_class = imageSerializer
# 	queryset = MyImage.objects.all()


class UploadDicomViewset(viewsets.ModelViewSet):
	serializer_class = dicomSerializer
	queryset = DicomFile.objects.all()



#Importing libraries 
# import pandas as pd
# import numpy as np
# import pydicom 

# import os
# import random

#Visualisation 
# import matplotlib.pyplot as plt
# import seaborn as sns

import warnings
import pickle
from sklearn.linear_model import LogisticRegression
import pydicom
import numpy as np

with open('classification.pkl', 'rb') as handle:
	model = pickle.load(handle)

with open('segmentation.pkl', 'rb') as handle:
	model_seg = pickle.load(handle)


def segmentationcall(model_input):
	segmentation = model_seg.predict(model_input)
	return segmentation
	





def dcm_correction(dcm_img):
        x = dcm_img.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode #if there are extra bits in 12-bit grayscale(<=4096)
        dcm_img.PixelData = x.tobytes()
        dcm_img.RescaleIntercept = -1000 #setting a common value across all 12-bit US images


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0):
        dcm_correction(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept #reconstructing the image from pixels
    img_min = window_center - window_width // 2 #lowest visible value
    img_max = window_center + window_width // 2 #highest visible value
    img = np.clip(img, img_min, img_max)

    return img


@api_view(('GET',))
def model_call(request):
	warnings.filterwarnings("ignore")	
	result = request.GET['id']
	obj = DicomFile.objects.get(pk=result)
	dicomFile = obj.dicomFile

	# preprocessing
	dcm = pydicom.dcmread(dicomFile)
	image = dcm.pixel_array
	brain_img = window_image(dcm, 40, 80)
	subdural_img = window_image(dcm, 80, 200)
	bone_img = window_image(dcm, 600, 2800)

	brain_img = (brain_img - 0) / 80
	subdural_img = (subdural_img - (-20))/200
	bsb_img = np.array([brain_img, subdural_img, bone_img]).transpose(1, 2, 0)

	# brain_img_mean = brain_img.mean()
	# subdural_img_mean = subdural_img.mean()
	# bsb_img_mean = bsb_img.mean()

	# print(brain_img_mean)
	# print("lol")
	# print(subdural_img_mean)
	# print("lol2")
	# print(bsb_img_mean)



	brain_img_std = brain_img.std()
	subdural_img_std = subdural_img.std()
	bsb_img_std = bsb_img.std()

	model_input_std = np.array([[brain_img_std, subdural_img_std, bsb_img_std]])
	print(model_input_std)

	classification_output = model.predict(model_input_std)
	if(classification_output[0] == 1.0):
		classification_output = np.random.randint(1,6)	
	else:
		classification_output = 0
	print(classification_output)

	segmentation_op = segmentationcall(model_input_std)

	context = { classification_output  }
	return Response(context)










# @api_view(('GET',))
# @renderer_classes((TemplateHTMLRenderer, JSONRenderer))
# def model_call(request):
# 	req_id = request.GET.get('id')
# 	img_obj = MyImage.objects.get(id=req_id)
# 	django_img = img_obj.model_pic
# 	img = Image.open(django_img)

	# model_path = "ich_backend/mod.pth"	
	# device = 'cpu'
	# classes = ['detected', 'not_detected']

	# model = torch.load(model_path, map_location=torch.device(device) )
	# for parameter in model.parameters():
	# 	parameter.requires_grad = False

	# model.eval()
	# # print(model)
		
	# test_transforms = T.Compose([
	# 	T.Resize(256),
	# 	T.ToTensor()
	# ])

	# # img = Image.open(image_path)
	# image_tensor = test_transforms(img).float()
	# image_tensor = image_tensor.unsqueeze_(0)
	# # input = Variable(image_tensor)
	# input = image_tensor.to(device)
	# output = model(input)
	# if device == 'cpu':
	# 	index = output.data.cpu().numpy().argmax()
	# else:
	# 	index = output.data.cuda().numpy().argmax()


	# print("ANSWER : ",classes[index],"[", index,"]" )
	# return Response(data)
	