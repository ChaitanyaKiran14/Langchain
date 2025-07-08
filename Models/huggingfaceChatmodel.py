from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

envValue = os.getenv("HUGGING_FACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token= envValue
)

chat_model = ChatHuggingFace(llm=llm)
output = chat_model.invoke(input = "Who has most number of world championships in F1")
print(output.content)
