# -*- coding: utf-8 -*- 
from __future__ import unicode_literals
#django
from django.urls import path
#python
# - - - 
#gazepattern
from . import views

urlpatterns = [
	path('', views.HomeView.as_view(), name="home"),
	path('images/', views.ImagesView.as_view(), name="images"),
	path('clasificar/<int:image_id>/', views.ImageClasificarView.as_view(), name="clasificar")
]