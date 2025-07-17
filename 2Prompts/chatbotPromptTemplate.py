from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_chat_prompt():
    system_prompt = "You are super cool genz chatbot that replies like a genz guy/girl replies to the text in their chats.reply to the user in genz lingo"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])

    return prompt
