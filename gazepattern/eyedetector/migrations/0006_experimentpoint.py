# Generated by Django 2.2.5 on 2019-09-28 23:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eyedetector', '0005_experiment'),
    ]

    operations = [
        migrations.CreateModel(
            name='ExperimentPoint',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fixation_number', models.IntegerField()),
                ('x', models.DecimalField(decimal_places=30, max_digits=50)),
                ('y', models.DecimalField(decimal_places=30, max_digits=50)),
            ],
        ),
    ]
