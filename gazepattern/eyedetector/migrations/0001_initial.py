# Generated by Django 2.2.5 on 2019-09-14 22:30

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
                ('name', models.CharField(max_length=30)),
                ('description', models.CharField(max_length=320)),
                ('created_on', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]