from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from chatbotPromptTemplate import get_chat_prompt  

load_dotenv()
huggingFaceEnvValue = os.getenv("HUGGING_FACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceEnvValue,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

chat_model = ChatHuggingFace(llm=llm)

chatHistory = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Prepare prompt using your system template
    prompt_template = get_chat_prompt()
    messages = prompt_template.format_messages(user_input=user_input)

    # Send message to model
    output = chat_model.invoke(messages)
    print("AI:", output.content)

    chatHistory.append({"user": user_input, "ai": output.content})
