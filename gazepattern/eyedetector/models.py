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

	@property
	def get_x(self):
		x = self.x0 if self.x0 < self.x1 else self.x1
		return x / self.image.image.width
	
	@property
	def get_y(self):
		y = self.y0 if self.y0 < self.y1 else self.y1
		return y / self.image.image.height

	@property
	def width(self):
		width = abs(self.x0 - self.x1)
		percent = width / self.image.image.width
		return percent

	@property
	def height(self):
		height = abs(self.y0 - self.y1)
		percent = height / self.image.image.height
		return percent


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
	phi = models.CharField(max_length=500, blank=True, null=True)
	created_on = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return u"{} - {}".format(self.name, self.description)


class ExperimentPoint(models.Model):

	experiment = models.ForeignKey(Experiment, related_name='points', on_delete=models.CASCADE)
	fixation_number = models.IntegerField()
	x = models.DecimalField(max_digits=50, decimal_places=30)
	y = models.DecimalField(max_digits=50, decimal_places=30) 

	@property
	def get_x(self):
		return self.x / self.experiment.image.image.width

	@property
	def get_y(self):
		return self.y / self.experiment.image.image.height


class ExperimentFunction(models.Model):
	experiment = models.ForeignKey(Experiment, blank=True, null=True, on_delete=models.CASCADE, related_name="experimentfunctions")
	function = models.TextField(max_length=200, help_text="Funcion en formato de codigo python")

	def __str__(self):
		return self.function

class ExperimentPointCSV(models.Model):
	file = models.FileField(upload_to="csv/")