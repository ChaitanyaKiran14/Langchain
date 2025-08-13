from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os
load_dotenv()
envValue = os.getenv("HUGGING_FACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token= envValue
)

model1 = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
model2 = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template= """You are the cricket pitch report expert, based on the following match information ${match_info} , provide a detailed pitch report for the match.""",
    input_variables=['match_info'],
    validate_template=True
)

prompt2 = PromptTemplate(
    template= """"Based on the tosses won by the teams in the past at the pitch ${pitch_name} provide analysis of the match outcome""",
    input_variables=['pitch_name'],
    validate_template=True
)

prompt3 = PromptTemplate(
    template= """Based on the pitch report ${pitch_report} and toss analysis ${toss_analysis} provide a detailed prediction of the match outcome""",
    input_variables=['pitch_report', 'toss_analysis'],
    validate_template=True
)

parser = StrOutputParser()

parallel_chain  = RunnableParallel(
    pitch_report = prompt1 | model1 | parser,
    toss_analysis = prompt2 | model2 | parser
)

final_chain = parallel_chain | prompt3 | model1 | parser

output = final_chain.invoke({
    "match_info": "SouthAfrica vs Australia, WTC Final, 2025, Day 1, Lords stadium, England",
    "pitch_name": "Lords"
})
print(output)
final_chain.get_graph().print_ascii()



 
