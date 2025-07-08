from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro",  google_api_key=os.getenv("GEMINI_API_KEY"))
output = model.invoke("Most powerful alien among ben 10 all aliens")

print("Gen from" + output.content)