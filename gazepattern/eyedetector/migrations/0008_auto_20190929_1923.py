# Generated by Django 2.2.5 on 2019-09-29 19:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eyedetector', '0007_experimentpoint_experiment'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='functions',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='relations',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AddField(
            model_name='experiment',
            name='result',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]