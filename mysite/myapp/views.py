from django.shortcuts import render
from django.http import HttpResponse

from joblib import load

model=load("ml38.joblib")

def home(request):
    return render(request,"home.html")

def add(request):
    return render(request,"a.html")

def pred(request):
    e=request.POST['exp']
    inp=[[float(e)]]
    yp=model.predict(inp)
    return HttpResponse(yp)