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
	form_class = ImageForm

	def get_context(self, request):
		data = request.GET if request.method == "GET" else request.POST
		context = {
			'form': self.form_class,
			'form_data': data,
		}
		return context

	def post(self, request):
		context = self.get_context(request)
		form = self.form_class(request.POST, request.FILES)
		if form.is_valid():
			form.save()
		context['form'] = form
		return render(request, self.template, context)