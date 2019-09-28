# -*- coding: utf-8 -*-
from django.contrib import admin
from .models import Experiment, ExperimentPoint, Image, ImageRectangle

admin.site.register([Experiment, ExperimentPoint, Image, ImageRectangle])

