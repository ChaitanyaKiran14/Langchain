from pydantic import BaseModel, Field
from typing_extensions import Optional, Literal
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": { "type": "string" },
      "description": "List all the key themes discussed in the review. Output as an array of short strings."
    },
    "summary": {
      "type": "string",
      "description": "Summarize the review briefly in 1–2 sentences."
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Sentiment of the review. Only return 'pos' or 'neg'."
    },
    "pros": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List the pros as individual strings in an array. If no pros, return null."
    },
    "cons": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List the cons as individual strings in an array. If no cons, return null."
    },
    "name": {
      "type": ["string", "null"],
      "description": "Name of the reviewer. Return null if not provided."
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

structred_output = model.with_structured_output(json_schema)
result = structred_output.invoke("""my phone is lacking good camera and i found there is black spot in the display""")
print(result)