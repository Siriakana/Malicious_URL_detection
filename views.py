from django.views import View
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserRegisterModel

#========================================================================================================
class UserRegistrationView(View):
    template_name = 'admins/register.html'
    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)
    def post(self, request, *args, **kwargs):
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['pswd1']

        # Check if the username already exists
        if UserRegisterModel.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return render(request, self.template_name)

        # Check if the email already exists
        if UserRegisterModel.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists')
            return render(request, self.template_name)

        # If both username and email are unique, create a new user
        user = UserRegisterModel.objects.create(
            username=username,
            email=email,
            password=password
        )
        user.save()
        messages.success(request, 'User registered successfully')

        return redirect('register')   

#==============================================================================================================

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('username')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return redirect('adminhome')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'admins/adminlogin.html', {})


#============================================================================================

 
def UserLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('username')
        password = request.POST.get('pswd')
        print("User ID is = ", usrid)
        try:
            user = UserRegisterModel.objects.get(username=usrid, password = password )
            if user.is_active:    
                return render(request, 'users/userbase.html')
            else:
                messages.success(request, 'User is not activated, Permissions denined..')
                return render(request, 'admins/login.html', {})
        except:
            messages.error(request,'Invalid Crediatials')
            return render(request, 'admins/login.html', {})
            
    return render(request, 'admins/login.html') 

#====================================================================================================

def AdminHomePage(request):
    data = UserRegisterModel.objects.all()
    return render(request, 'admins/AdminHome.html', {'data':data})

#===================================================================================================

def UserActivateFunction(request, pk):
    user = UserRegisterModel.objects.get(id=pk)
    user.is_active = True
    user.save()
    return redirect(AdminHomePage)

#=======================================================================================================

def UserDeactivateFunction(request, pk):
    user = UserRegisterModel.objects.get(id=pk)
    user.is_active = False
    user.save()
    return redirect(AdminHomePage)

#==========================================================================================================

def home1(request):
    return render(request,'home.html') 