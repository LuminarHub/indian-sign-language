from django.db import models

# Create your models here.

class Registration(models.Model):
    username=models.CharField(unique=True,max_length=100,null=True,blank=True)
    Email=models.EmailField(unique=True,max_length=100,null=True,blank=True)
    Password=models.CharField(max_length=8,null=True,blank=True)
    Confirm_Password=models.CharField(max_length=8,null=True,blank=True)
    audio=models.FileField(upload_to='audios',null=True,blank=True)

   
    
    
class Audio(models.Model):
    user = models.ForeignKey(Registration,on_delete=models.CASCADE,related_name="user_add")
    audio_new =models.FileField(upload_to='audios_generated',null=True,blank=True)
