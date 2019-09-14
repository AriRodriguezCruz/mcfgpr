# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.db import models


class Image(models.Model):
	image = models.ImageField(upload_to='images/')
	name = models.CharField(max_length=30)
	description = models.CharField(max_length=320)
	created_on = models.DateTimeField(auto_now_add=True)
