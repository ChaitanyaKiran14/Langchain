from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
output = model.invoke("Hello, world!")
print(output.content)