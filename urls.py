"""Phishing_Url URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from admins import views
from users import views as UserViews
from admins.views import UserRegistrationView
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home1, name='home1'),
    path('userregistration', UserRegistrationView.as_view(), name='register'),
    path('adminlogin', views.AdminLoginCheck, name='adminlogin'),
    path('userlogin', views.UserLoginCheck, name='login'),

    path('adminhome', views.AdminHomePage, name='adminhome'),
    path('useractivate/<int:pk>/', views.UserActivateFunction, name='useractivate'),
    path('userdeactivate/<int:pk>/', views.UserDeactivateFunction, name='deactivate'),


    path('userhome', UserViews.UserHome, name='userhome'),
    path('training',UserViews.training, name='training' ),
    path('prediction', UserViews.prediction, name='prediction'),
]
