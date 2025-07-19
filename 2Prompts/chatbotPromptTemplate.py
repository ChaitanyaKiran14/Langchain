from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
def get_chat_prompt():
    system_prompt = "You are super cool genz friend that replies like a genz guy/girl replies to the text in their chats.reply to the user in genz lingo. You remember everything from the user input and reply the user back"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])
    return prompt

prompt2  = ChatPromptTemplate({
    SystemMessage(content="you are ${domain} expert"),
    HumanMessage(content="Explain in simple terms about ${topic}")
})

val = prompt2.invoke({'domain': 'chess', 'topic' :'opening'})
print(val)