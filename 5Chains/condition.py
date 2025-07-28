from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.schema.runnable import RunnableParallel, RunnableBranch
from pydantic import BaseModel, Field
from typing_extensions import Literal

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print(GOOGLE_API_KEY)
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
class FeedbackSentiment(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description="Given feedback from the user, understand the sentiment of the feedback")

structured_output = model.with_structured_output(FeedbackSentiment)
result = structured_output.invoke("""excellent product""")
print(result.sentiment)


if result.sentiment == 'negative':
    print("The feedback is negative.")
elif result.sentiment == 'positive':
    print("The feedback is positive.")