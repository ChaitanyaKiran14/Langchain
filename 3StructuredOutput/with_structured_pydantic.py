from pydantic import BaseModel, Field
from typing_extensions import Optional, Literal
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


class Review(BaseModel):
     key_themes : list[str] = Field(description= "Write down all the key themes discussed in the review in a list")
     summary    : str = Field(description="A brief summary of the review")
     sentiment  : Literal["pos", "neg" , "meutral"] = Field(description="Return sentiment of the review either negative, positive or neutral")
     pros       :Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
     cons       : Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
     name       : Optional[str] = Field(default=None,description="Write the name of the reviewer")

structred_output = model.with_structured_output(Review)
result = structred_output.invoke("""my phone is lacking good camera and i found there is black spot in the display""")
print(result)