# Generated by Django 3.1.1 on 2020-10-29 08:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0009_auto_20201009_1937'),
    ]

    operations = [
        migrations.RenameField(
            model_name='post',
            old_name='body',
            new_name='business',
        ),
        migrations.AddField(
            model_name='post',
            name='cons',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='post',
            name='pros',
            field=models.TextField(default=''),
        ),
    ]
