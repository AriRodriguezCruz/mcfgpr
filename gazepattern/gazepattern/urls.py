# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(("api.urls", "api", ), namespace="api" )),
    path('', include('eyedetector.urls'))
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)