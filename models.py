from django.db import models

class UserRegisterModel(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=100)
    is_active = models.BooleanField(default=False)
    

    def __str__(self):
        return self.username
    