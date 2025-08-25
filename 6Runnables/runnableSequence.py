#from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
##chain = prompt1| model | parser | prompt2 | model| parser

print(chain.invoke({'topic':'AI'}))
chain.get_graph().print_ascii()

