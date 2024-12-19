from django.shortcuts import render, redirect

from django.contrib.auth.hashers import check_password
from django.views import View

from ..models.employee import *
from django.http import JsonResponse

import json

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

# DON'T USE THAT IN PROD
@method_decorator(csrf_exempt, name='dispatch') 
class EmployeeRegistration(View):
    
    def post(self, request):
        try:
            print("Raw request body:", request.body) 
            
            data = json.loads(request.body)

            employee_name = data.get('employeeFullname')
            employee_email = data.get('employeeEmail')
            employee_password = data.get('employeePassword')
            
            employees = Employee.get_all_employees()
            
            if employees.filter(email=employee_email).exists():
                return JsonResponse({'error': 'Email already exists.'}, status=400)
            
            # What should I do about the password? it's not being added
            
            
            employee = Employee(name=employee_name, email=employee_email)
            employee.set_password(raw_password=employee_password)
            
            employee.save()
                    
        except Exception as err:
            print('Error: ', err)
            return JsonResponse({'error': 'An error occured while registering the product: '}, status=500)
            
        return redirect('index')

@method_decorator(csrf_exempt, name='dispatch') 
class EmployeeLogin(View):
    
    def post(self, request):
        try:
            print("Raw request body:", request.body) 
            
            data = json.loads(request.body)

            employee_email = data.get('employeeEmail')
            employee_password = data.get('employeePassword')
            
            employee = Employee.get_employee_by_email(email=employee_email)
            
            password_exists = Employee.check_password(self=employee, raw_password=employee_password)
            
            if employee and password_exists:
                # Store employee ID in the session
                refresh = RefreshToken.for_user(employee)
                
                return JsonResponse({'success': 'Logged in successfully.'}, status=200)
            
            return JsonResponse({'error': 'Email already exists.'}, status=400)
            
            # I'm relying on the fact that the password field is directly checked by the hash I create.
            
                
        except Exception as err:
            print('Error: ', err)
            return JsonResponse({'error': 'An error occured while registering the product: '}, status=500)

class EmployeeDetails(View):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        print("Session data after login:", request.session.items())
        employee_id = request.session.get('employee_id')
        if not employee_id:
            return JsonResponse({'error': 'You are not logged in.'}, status=401)
        
        employee = Employee.get_employee_by_id(employee_id=employee_id).first()
        
        if employee:
            return JsonResponse({
                'employee_id': employee.employee_id,
                'name': employee.name,
                'email': employee.email
            }, status=200)
        return JsonResponse({'error': 'Employee not found.'}, status=404)
