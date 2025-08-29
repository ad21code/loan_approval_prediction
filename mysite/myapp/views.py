from django.shortcuts import render
from django.http import HttpResponse

from joblib import load

model=load("loan.joblib")

def home(request):
    return render(request,"home.html")

def add(request):
    return render(request,"a.html")

def pred(request):
    gender = int(request.POST.get('Gender'))
    married = int(request.POST.get('Married'))
    dependents = int(request.POST.get('Dependents'))
    education = int(request.POST.get('Education'))
    self_employed = int(request.POST.get('Self_Employed'))
    applicant_income = float(request.POST.get('ApplicantIncome'))
    coapplicant_income = float(request.POST.get('CoapplicantIncome'))
    loan_amount = float(request.POST.get('LoanAmount'))
    loan_amount_term = float(request.POST.get('Loan_Amount_Term'))
    credit_history = int(request.POST.get('Credit_History'))
    property_area = int(request.POST.get('Property_Area'))
    inp=[[gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area]]
    yp=model.predict(inp)

    return render(request,"pred.html",{'myresult':yp[0]})