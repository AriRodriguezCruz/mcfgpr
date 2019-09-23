# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.db import models


class Image(models.Model):
	image = models.ImageField(upload_to='images/')
	name = models.CharField(max_length=30)
	description = models.CharField(max_length=320)
	created_on = models.DateTimeField(auto_now_add=True)

class ImageRectangle(models.Model):
	image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name='rectangles')
	x0 = models.DecimalField(max_digits=9, decimal_places=4)
	x1 = models.DecimalField(max_digits=9, decimal_places=4)
	y0 = models.DecimalField(max_digits=9, decimal_places=4)
	y1 = models.DecimalField(max_digits=9, decimal_places=4)
	name = models.CharField(max_length=50)

class XYPupilFrame(models.Model):
	x = models.DecimalField(max_digits=50, decimal_places=30)
	y = models.DecimalField(max_digits=50, decimal_places=30) 