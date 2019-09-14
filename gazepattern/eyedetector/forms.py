# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django import forms
#gazepattern
from utils.widgets import TextWidget

class ImageForm(forms.Form):
	name = forms.CharField(required=True, widget=TextWidget(label="Nombre"))
	description = forms.CharField(required=True, widget=TextWidget(label="Descripcion"))
	image = forms.ImageField(required=True, label='Subir imagen')
