from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
envValue = os.getenv("HUGGING_FACE_API_KEY")


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=envValue
)

prompt_template = PromptTemplate.from_template("hey there how's the weather in {place}")
userInput = str(input("Enter your location :"))

prompt = prompt_template.format(place = userInput)
response = llm.invoke(prompt)
print(response)

