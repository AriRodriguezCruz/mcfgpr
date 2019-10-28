# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.shortcuts import render
#python
# - - -
#gazepattern
from utils.views import ApiView
from eyedetector.models import Image, Experiment


class RectanglesView(ApiView):

	def json(self, request, image_id):
		image = Image.objects.get(pk=image_id)
		rectangles = image.rectangles.all()
		results = [
			{
				"id": rectangle.pk,
				"name": rectangle.name,
				"x": rectangle.get_x,
				"y": rectangle.get_y,
				"width": rectangle.width, 
				"height": rectangle.height,
			} for rectangle in rectangles
		]

		response = {"results": results}

		return response


class PointsView(ApiView):

	def json(self, request, experiment_id):
		experiment = Experiment.objects.get(pk=experiment_id)
		points = experiment.points.all().order_by("fixation_number")
		results = [
			{
				"id": point.pk,
				"x": point.get_x,
				"y": point.get_y,
			} for point in points
		]

		response = {"results": results}

		return response

