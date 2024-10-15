from django.db import models

class CustomUser(models.Model):
    username = models.CharField(max_length=30, unique=True)
    password = models.CharField(max_length=128)  # Store hashed passwords, not plain text

    def __str__(self):
        return self.username
