# Generated by Django 4.0 on 2022-03-09 09:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ich_backend', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='DicomFile',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('dicomFile', models.FileField(upload_to='')),
            ],
        ),
    ]
