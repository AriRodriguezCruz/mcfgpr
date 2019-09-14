# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.views import View
from django.shortcuts import render
#python
# - - -

class BaseView(View):
	template = None

	def get_context(self, request, *args):
		context = {}
		return context

	def get(self, request, *args):
		context = self.get_context(request, *args)
		return render(request, self.template, context)

	def post(self, request, *args):
		context = self.get_context(request, *args)
		return context