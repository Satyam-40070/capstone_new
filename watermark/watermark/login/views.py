from django.shortcuts import render
from django.http import HttpResponse
from .models import CustomUser

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if CustomUser.objects.filter(username=username).exists():
            return HttpResponse("Username already exists.")
        user = CustomUser.objects.create(username=username, password=password)
        return HttpResponse("Registration successful.")
    return render(request, 'login/register.html')

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            user = CustomUser.objects.get(username=username)
            if user.password == password:
                return HttpResponse("Login successful.")
            else:
                return HttpResponse("Invalid credentials.")

        except Exception:
            return HttpResponse("User does not exist")

    return render(request, 'login/login.html')
