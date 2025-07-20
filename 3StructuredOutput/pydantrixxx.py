from pydantic import BaseModel
from typing_extensions import Optional


class Student(BaseModel):
    name : str = "kiran"
    age : Optional[int]

new_student =  {}
student = Student(**new_student)
print(student)
