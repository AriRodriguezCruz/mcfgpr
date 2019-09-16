# Generated by Django 2.2.5 on 2019-09-16 01:41

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('eyedetector', '0002_imagerectangle'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagerectangle',
            name='image',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='rectangles', to='eyedetector.Image'),
            preserve_default=False,
        ),
    ]
