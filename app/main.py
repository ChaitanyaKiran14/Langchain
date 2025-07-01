from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
app = FastAPI()



@app.get("/")
def read_root():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro",  google_api_key=os.getenv("GEMINI_API_KEY"))
    output = model.invoke("Hey wassup")
    return {"message": output.content}