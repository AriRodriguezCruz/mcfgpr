# Generated by Django 2.2.5 on 2019-09-29 21:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eyedetector', '0008_auto_20190929_1923'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='functions',
            field=models.CharField(blank=True, max_length=5000, null=True),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='relations',
            field=models.CharField(blank=True, max_length=5000, null=True),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='result',
            field=models.CharField(blank=True, max_length=5000, null=True),
        ),
    ]
