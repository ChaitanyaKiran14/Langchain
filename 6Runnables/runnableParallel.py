from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel
import os
from langchain.chat_models import init_chat_model
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)



parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print("Tweet : ", result["tweet"])
print("Linkedin Post : ", result["linkedin"])

parallel_chain.get_graph().print_ascii()