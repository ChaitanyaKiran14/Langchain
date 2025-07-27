from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2  | model|  parser
result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)
