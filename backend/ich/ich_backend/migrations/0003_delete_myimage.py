# Generated by Django 4.0 on 2022-03-09 09:57

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ich_backend', '0002_dicomfile'),
    ]

    operations = [
        migrations.DeleteModel(
            name='MyImage',
        ),
    ]