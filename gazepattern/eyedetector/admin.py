# -*- coding: utf-8 -*-
#django
from django.contrib import admin
from django.db import transaction 
#python
import csv
from decimal import Decimal
#gazepattern
from .models import Experiment, ExperimentPoint, Image, ImageRectangle, ExperimentPointCSV, ExperimentFunction

@transaction.atomic
def procesar(modeladmin, request, queryset):
    for query in queryset:
        file = query.file
        with open(file.path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            rows = [row for row in csv_reader if len(row)]
            for row in rows:
                experiment_id = int(row[0])
                fixation_number = int(row[1])
                x = Decimal(row[2])
                y = Decimal(row[3])

                experiment = Experiment.objects.get(pk=experiment_id)
                experiment_point = ExperimentPoint()
                experiment_point.experiment = experiment
                experiment_point.fixation_number = fixation_number
                experiment_point.x = x
                experiment_point.y = y
                experiment_point.save()

procesar.short_description = "Procesar CSV para generar experiments points"


class ExperimentPointCSVAdmin(admin.ModelAdmin):
    list_display = ['id', 'file']
    ordering = ['id']
    actions = [procesar, ]


class ExperimentPointAdmin(admin.ModelAdmin):
    list_display = ['id', 'experiment_id', 'fixation_number', 'x', 'y']
    ordering = ['id']
    search_fields = ["experiment__id"]


class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    ordering = ['id']


class ExperimentAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'description']
    ordering = ['id']


class ImageRectangleAdmin(admin.ModelAdmin):
    list_display = ['id', 'image_id','name']
    ordering = ['id']
    search_fields = ['image__id']


class ExperimentFunctionAdmin(admin.ModelAdmin):
    list_display = ['id', 'experiment_id', 'function']
    ordering = ['id']
    search_fields = ['experiment__id']


admin.site.register(ExperimentPointCSV, ExperimentPointCSVAdmin)
admin.site.register(ExperimentPoint, ExperimentPointAdmin)
admin.site.register(Image, ImageAdmin)
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(ImageRectangle, ImageRectangleAdmin)
admin.site.register(ExperimentFunction, ExperimentFunctionAdmin)