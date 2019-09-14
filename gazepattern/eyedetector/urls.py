# -*- coding: utf-8 -*- 
from __future__ import unicode_literals
#django
from django.urls import path
#python
# - - - 
#gazepattern
from . import views

urlpatterns = [
	path('', views.Home.as_view(), name="home")
]
