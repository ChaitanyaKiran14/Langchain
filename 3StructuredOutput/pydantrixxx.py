from pydantic import BaseModel, EmailStr,Field
from typing_extensions import Optional


class Student(BaseModel):
    name : str = "kiran"
    age : Optional[int] = 22
    email : EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

new_student =  {"name": "Hey Chaitu", "email" : "chaitu@gmail.com", "cgpa" : 9}
student = Student(**new_student)
print(student)

student_dict = dict(student)
print(student_dict['age'])
student_json = student.model_dump_json()
print(student_json)