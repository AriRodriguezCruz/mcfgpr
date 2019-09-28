# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render, get_object_or_404
from django.db import transaction
#python 
# - - -
#gazepattern
from utils.views import BaseView
from eyedetector.forms import ImageForm, MakeExperimentForm
from eyedetector.models import Image, Experiment
from gui.application import Application
from gui.experiment import CheckCamera, Training, MakeExperiment

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
		return context

	def get(self, request, image_id):
		context = self.get_context(request, image_id)
		image = context.get('image')
		app = Application(image.image.file.name, image)
		return render(request, self.template, context)


class ExperimentView(BaseView):

	template = "eyedetector/experiment.html"
	form_class = MakeExperimentForm

	def get_context(self, request):
		context = {
			"images": Image.objects.all(),
			'form': self.form_class
		}
		return context


class CheckCameraView(BaseView):
	template = "generic_template.html"

	def get(self, request,  *args):
		CheckCamera()
		return render(request, self.template, locals())


class TrainView(BaseView):
	template = "generic_template.html"

	def get(self, request, *args):
		Training()
		return render(request, self.template, locals())


class MakeExperimentView(BaseView):

	template = "generic_template.html"

	@transaction.atomic
	def post(self, request, image_id):
		form = MakeExperimentForm(request.POST)
		if form.is_valid():
			image = get_object_or_404(Image, pk=image_id)
			experiment = Experiment()
			experiment.name = request.POST.get('name')
			experiment.description = request.POST.get('description')
			experiment.image = image
			experiment.save()
			MakeExperiment(image.image.file.name)
		return render(request, self.template, {})					