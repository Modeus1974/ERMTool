# Generated by Django 3.1.1 on 2020-10-09 07:36

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('blog', '0007_auto_20201009_1452'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='votes',
            field=models.ManyToManyField(related_name='voted_posts', through='blog.Vote', to=settings.AUTH_USER_MODEL),
        ),
    ]
