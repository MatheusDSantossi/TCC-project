from django.contrib.auth.hashers import PBKDF2PasswordHasher
from django.db import models
from django.shortcuts import render
from django.contrib.postgres.fields import ArrayField
from project_ecommerce_AI import generate_synonymous, embed_fn, pre_process_prompt_grammatical, fetch_products_alike

class Employee(models.Model):

    employee_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, default='')
    email = models.CharField(max_length=255, default='')
    password = models.CharField(max_length=255, default='')
    password_hash = models.CharField(max_length=255, default='')
    
    # def set_password(self, raw_password):
    # self.password_hash = make_password(raw_password)

    # def check_password(self, raw_password):
    #     return check_password(raw_password, self.password_hash)
    
    def set_password(self, raw_password):
        # Using a secure password hashing algorithm
        hasher = PBKDF2PasswordHasher()
        salt = hasher.salt()
        password_hash = hasher.encode(raw_password, salt)
        self.password_hash = password_hash
           
    def check_password(self, raw_password):
        return PBKDF2PasswordHasher().verify(raw_password, self.password_hash)
    
    def get_all_employees():
        return Employee.objects.all()
    
    @staticmethod
    def get_employee_by_id(employee_id):
        return Employee.objects.get(employee_id=employee_id)
    
    @staticmethod
    def get_employee_by_email(email):
        return Employee.objects.get(email=email)

    class Meta:
        db_table = 'employees'

    def __str__(self):
        return self.name
