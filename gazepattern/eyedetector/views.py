# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
#python 
# - - -
#gazepattern
from utils.views import BaseView
from eyedetector.forms import ImageForm

class Home(BaseView):
	template = "eyedetector/home.html"

class Images(BaseView):
	template = "eyedetector/images.html"

	def get_context(self, request):
		data = request.GET if request.method == "GET" else request.POST
		context = {
			'form': ImageForm,
			'form_data': data,
		}
		return context