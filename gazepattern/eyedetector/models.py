# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.db import models


class Image(models.Model):
	image = models.ImageField(upload_to='images/')
	name = models.CharField(max_length=30)
	description = models.CharField(max_length=320)
	created_on = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return "{} - {}".format(self.pk , self.name)

class ImageRectangle(models.Model):
	image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name='rectangles')
	x0 = models.DecimalField(max_digits=9, decimal_places=4)
	x1 = models.DecimalField(max_digits=9, decimal_places=4)
	y0 = models.DecimalField(max_digits=9, decimal_places=4)
	y1 = models.DecimalField(max_digits=9, decimal_places=4)
	name = models.CharField(max_length=50)

	def __str__(self):
		return '<a href="/admin/eyedetector/image/{}/">{}<a/> -- {}'.format(self.image.pk, self.image.pk, self.name)


class XYPupilFrame(models.Model):
	x = models.DecimalField(max_digits=50, decimal_places=30)
	y = models.DecimalField(max_digits=50, decimal_places=30) 


class Experiment(models.Model):
	name = models.CharField(max_length=50)
	description = models.CharField(max_length=350, blank=True, null=True)
	image = models.ForeignKey(Image, related_name='experiments', on_delete=models.CASCADE)
	functions = models.CharField(max_length=5000, blank=True, null=True)
	relations = models.CharField(max_length=5000, blank=True, null=True)
	result = models.CharField(max_length=5000, blank=True, null=True)

	def __str__(self):
		return u"{} - {}".format(self.name, self.description)


class ExperimentPoint(models.Model):

	experiment = models.ForeignKey(Experiment, related_name='points', on_delete=models.CASCADE)
	fixation_number = models.IntegerField()
	x = models.DecimalField(max_digits=50, decimal_places=30)
	y = models.DecimalField(max_digits=50, decimal_places=30) 