from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

 
@tool
def sum(a: int, b : int)-> int:
    """given two numbers it adds them"""
    return a +b
@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

@tool
def powerup(a : int) -> int :
    """given an integer it raises the power to two"""
    return a**2


query = HumanMessage('can you add 10 to 12')
messages = [query]
#print(messages)

llm_with_tools = llm.bind_tools([sum,multiply,powerup ])
result = llm_with_tools.invoke(messages)

print(result.tool_calls[0])
messages.append(result)

tool_name = result.tool_calls[0]["name"]
tool_args = result.tool_calls[0]["args"]


print("Tool Selected:", tool_name)
print("Tool Args:", tool_args)

#print(messages)
if tool_name == "powerup":
    tool_result = powerup.invoke(tool_args)
elif tool_name == "multiply":
    tool_result = multiply.invoke(tool_args)
elif tool_name == "sum":
    tool_result = sum.invoke(tool_args)
else:
    tool_result = "Unknown tool"

tool_msg = ToolMessage(content=str(tool_result), tool_call_id=result.tool_calls[0]["id"])
messages.append(tool_msg)

print("Conversation:", messages)
