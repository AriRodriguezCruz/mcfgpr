# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
#python 
# - - -
#gazepattern
from utils.views import BaseView

class Home(BaseView):
	template = "eyedetector/home.html"

class Images(BaseView):
	template = "eyedetector/images.html"