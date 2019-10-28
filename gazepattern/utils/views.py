# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
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
		return render(request, self.template, context)


class ApiView(BaseView):

	def json(self, request, *args):
		response = {}
		return response

	def get(self, request, object_id, **kwargs):
		json = self.json(request, object_id)
		return JsonResponse(json)
		
	def post(self, reuest, *args, **kwargs):
		return get(self, request, *args)