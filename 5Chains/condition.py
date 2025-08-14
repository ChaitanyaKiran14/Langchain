from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing_extensions import Literal

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print(GOOGLE_API_KEY)
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
class FeedbackSentiment(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description="Given feedback from the user, understand the sentiment of the feedback")

structured_output = model.with_structured_output(FeedbackSentiment)

def detect_sentiment_with_feedback(feedback : str):
    sentiment_result = structured_output.invoke(feedback)
    return{
        "feedback": feedback,
        "sentiment": sentiment_result.sentiment

    }

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="""Write an smart and simple response to the positive feedback {feedback} of the user""",
    input_variables=['feedback']

)

prompt2 = PromptTemplate(
    template="""Write an smart and simple response to the negative feedback{feedback} of the user""",
    input_variables=['feedback']
)
positiveChain = prompt1 | model | parser
negativeChain = prompt2 | model | parser


branchChain = RunnableBranch(
    (lambda x:x['sentiment'] =='positive', positiveChain ),
    ( lambda x:x['sentiment']=='negative' , negativeChain ),
    RunnableLambda(lambda x: "couldn't find sentiment")
    
)


def feedback_pipeline(feedback):
    sentiment_data = detect_sentiment_with_feedback(feedback)
    return branchChain.invoke(sentiment_data)



print(feedback_pipeline("your store got the best of both worlds smartphones"))
