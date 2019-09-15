# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
#python 
import tkinter as tk
from tkinter import Label,Tk
from PIL import Image, ImageTk
import PIL.Image
#gazepattern
from utils.views import BaseView
from eyedetector.forms import ImageForm
from eyedetector.models import Image

class App(object):
	"""docstring for App"""
	def __init__(self, arg):
		self.arg = arg
		

class HomeView(BaseView):
	template = "eyedetector/home.html"


class ImagesView(BaseView):
	template = "eyedetector/images.html"
	form_class = ImageForm

	def get_context(self, request):
		data = request.GET if request.method == "GET" else request.POST
		context = {
			'form': self.form_class,
			'form_data': data,
			'images': Image.objects.all(),
		}
		return context

	def post(self, request):
		form = self.form_class(request.POST, request.FILES)
		if form.is_valid():
			form.save()
		context = self.get_context(request)
		context['form'] = form
		return render(request, self.template, context)


class ImageClasificarView(BaseView):
	template = "generic_template.html"

	def get_context(self, request, image_id):
		image = Image.objects.get(pk=image_id)
		context = {
			'image': image,
		}

	def get(self, request, image_id):
		context = self.get_context(request, image_id)
		root = tk.Tk()
		root.geometry( "1000x720" )
		#root.attributes("-fullscreen", True)
		app = App(root)
		root.mainloop()
		return render(request, self.template, context)