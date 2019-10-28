# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.shortcuts import render
#python
# - - -
#gazepattern
from utils.views import ApiView
from eyedetector.models import Image


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