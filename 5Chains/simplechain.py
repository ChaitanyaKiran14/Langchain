#using with_structured_output with pydantic models instead of output parsers
from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

prompt = PromptTemplate(
   input_variables=["name"],
   template="""
You are a knowledgeable assistant. Given the name "{name}", infer the full name, height and birth year based on publicly known information about the person. 
If the person is a celebrity or public figure, use your knowledge. 
Respond only with structured data for the fields: name, height, and birth_year.
"""
)
class Person(BaseModel):
    name : str = Field(description="Full name of the person")
    height : Optional[int] = Field(default=None, description="Height of the person")
    birth_year : Optional[int] = Field(default=None, description="Birth year of the person")
    
structured_output = model.with_structured_output(Person)

chain = prompt | structured_output
result = chain.invoke({"name": "heroine Katherine Langford"})
print(result)
chain.get_graph().print_ascii()