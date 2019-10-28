# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render, get_object_or_404, reverse, redirect
from django.db import transaction
#python 
# - - -
#gazepattern
from utils.views import BaseView
from eyedetector.forms import ImageForm, MakeExperimentForm, GenerateResultsForm
from eyedetector.models import Image, Experiment, ExperimentFunction, ImageRectangle
from gui.application import Application
from gui.experiment import CheckCamera, Training, MakeExperiment, GenerateResults

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
		return redirect(reverse("images"))


class ImageRectanglesView(BaseView):
	template = "eyedetector/canvas.html"

	def get_context(self, request, image_id):
		context = {}
		context["rectangles"] = ImageRectangle.objects.all()
		context['image'] = Image.objects.get(pk=image_id)
		return context

	def get(self, request, image_id):
		context = self.get_context(request, image_id)
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
		return redirect(reverse('experiment'))	


class TrainView(BaseView):
	template = "generic_template.html"

	def get(self, request, *args):
		Training()
		return redirect(reverse('experiment'))	


class MakeExperimentView(BaseView):

	template = "generic_template.html"

	def get(self, request, image_id):
		return redirect(reverse('experiment'))

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
			MakeExperiment(image.image.file.name, experiment)
		return redirect(reverse('experiment'))					


class ResultsView(BaseView):
	template = "eyedetector/results.html"
	form_class = GenerateResultsForm

	def get_context(self, request):
		context = {}
		context['experiments'] = Experiment.objects.all().order_by("-pk")
		context['form'] = self.form_class(experiment = Experiment.objects.all())
		context['error'] = request.GET.get("error", False)
		return context

	def post(self, request):
		context = self.get_context(request)
		form = self.form_class(request.POST)
		if form.is_valid():
			phi = form.data.get('phi')
		return render(request, self.template, context)


class MakeResultsView(BaseView):
	template = "eyedetector/results.html"
	form_class = GenerateResultsForm

	def get(self, request, experiment_id):
		return self.post(request, experiment_id)

	def post(self, request, experiment_id):
		form = self.form_class(request.POST)
		if form.is_valid():
			formula = form.data.get("phi")
			experiment = get_object_or_404(Experiment, pk=experiment_id)
			try:
				result_manager = GenerateResults(experiment, formula)
				relations, functions, result = result_manager.generate_result()
				experiment.relations = str(relations)
				experiment.functions = str(functions)
				experiment.result = str(result)
				experiment.phi = str(formula)
				experiment.save()
				if not str(formula) in [experimentfunction.function for experimentfunction in experiment.experimentfunctions.all()]:
					experiment_function = ExperimentFunction()
					experiment_function.function = str(formula)
					experiment_function.experiment = experiment
					experiment_function.save()
			except Exception as e:
				url = "{}{}".format(reverse('results'), r'?error=Ocurrio un error, verifique que su formula sea correcta'.replace(" ", r"%20"))
				return redirect(url)
		return redirect(reverse("results"))