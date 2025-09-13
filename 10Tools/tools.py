from langchain_core.tools import tool

@tool
def mywish(a: str , b:str):
    """grant wishes"""
    return f"As yout wish master , your wishes are {a}, {b}"

result = mywish.invoke({"a":"lambo", "b": "ferrari"})
print(result)