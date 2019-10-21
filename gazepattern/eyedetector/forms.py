# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django import forms
#gazepattern
from utils.widgets import TextWidget, QueryAsChoiceWidget
from .models import Image

class ImageForm(forms.Form):
	name = forms.CharField(required=True, max_length=30, widget=TextWidget(label="Nombre"))
	description = forms.CharField(required=True, max_length=320, widget=TextWidget(label="Descripcion"))
	image = forms.ImageField(required=True, label='Subir imagen')

	def save(self):
		data = self.clean()
		image = Image()
		image.image = data.get('image')
		image.name = data.get('name')
		image.description = data.get('description')
		image.save()
		print(image)
		return image


class MakeExperimentForm(forms.Form):
	name = forms.CharField(required=True, max_length=30, widget=TextWidget(label="Nombre"))
	description = forms.CharField(required=True, max_length=320, widget=TextWidget(label="Descripcion"))


class GenerateResultsForm(forms.Form):
	phi = forms.CharField(required=True, max_length=320, widget=TextWidget(label="Inserte una formula nueva.", required=True))

	def __init__(self, *args, **kwargs):
		experiment = kwargs.pop("experiment", False)
		super(GenerateResultsForm, self).__init__(*args, **kwargs)
		#if experiment:
			#queryset = experiment.experimentfunctions.all()
			#self.fields['phi_options'] = forms.IntegerField(required=True, widget=QueryAsChoiceWidget(queryset=queryset))
