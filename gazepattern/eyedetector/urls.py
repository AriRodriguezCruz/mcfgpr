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
	path('clasificar/<int:image_id>/', views.ImageClasificarView.as_view(), name="clasificar"),
	path("images/<int:image_id>/rectangles/", views.ImageRectanglesView.as_view(), name="rectangles"),
	path('experiment/', views.ExperimentView.as_view(), name="experiment"),
	path('checkcamera/', views.CheckCameraView.as_view(), name="check_camera"),
	path('train/', views.TrainView.as_view(), name="train"),
	path('makeexperiment/<int:image_id>', views.MakeExperimentView.as_view(), name="make_experiment"),
	path('results/', views.ResultsView.as_view(), name="results"),
	path('makeresults/<int:experiment_id>', views.MakeResultsView.as_view(), name="make_results"),
]
