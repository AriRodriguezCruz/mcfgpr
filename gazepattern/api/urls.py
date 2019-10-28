# -*- coding: utf-8 -*- 
from __future__ import unicode_literals
#django
from django.urls import path
#python
# - - - 
#gazepattern
from . import views

urlpatterns = [
	path("rectangles/<int:object_id>/", views.RectanglesView.as_view(), name="rectangles"),
	path("points/<int:object_id>/", views.PointsView.as_view(), name="points"),
]
