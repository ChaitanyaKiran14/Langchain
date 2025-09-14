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
def multiply(a: int) -> int:
  """Given a number multiply it by 2"""
  return a * 2

@tool
def powerup(a : int) -> int :
    """given an integer it raises the power to 2"""
    return a**2


#query = HumanMessage('First add 3 and 4 , then multiply the result by 2, then raise it to the power of two')
#messages = [query]
#print(messages)

llm_with_tools = llm.bind_tools([sum,multiply,powerup ])
#result = llm_with_tools.invoke(messages)
#print(result.tool_calls[0])
#messages.append(result)

#tool_name = result.tool_calls[0]["name"]
#tool_args = result.tool_calls[0]["args"]


# #print(messages)
# if tool_name == "powerup":
#     tool_result = powerup.invoke(tool_args)
# elif tool_name == "multiply":
#     tool_result = multiply.invoke(tool_args)
# elif tool_name == "sum":
#     tool_result = sum.invoke(tool_args)
# else:
#     tool_result = "Unknown tool"

# tool_msg = ToolMessage(content=str(tool_result), tool_call_id=result.tool_calls[0]["id"])
# messages.append(tool_msg)
# #print("Conversation:", messages)

# final_output =llm_with_tools.invoke(messages)
# print(final_output.content)


def run_with_tools(llm_with_tools, messages):
    for _ in range(5):  
       
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if not result.tool_calls:
            return result

        for tool_call in result.tool_calls:
            print(f"\n Step → Raw tool call: {tool_call}")
            
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "sum":
                tool_result = sum.invoke(tool_args)
            elif tool_name == "multiply":
                tool_result = multiply.invoke(tool_args)
            elif tool_name == "powerup":
                tool_result = powerup.invoke(tool_args)
            else:
                tool_result = "Unknown tool"

            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            messages.append(tool_msg)

    raise RuntimeError("Too many tool calls – possible infinite loop")


query = HumanMessage("First add 3 and 4 , then multiply the result by 2, then raise it to the power of two")
messages = [query]

final_response = run_with_tools(llm_with_tools, messages)
print("Final Answer:", final_response.content)