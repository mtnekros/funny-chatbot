from typing import Callable, Dict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama


def get_tools() -> Dict[str, BaseTool]:
    """Return tools in a dictionary."""

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: first integer
            b: second integer

        """
        return a + b

    @tool
    def get_weather(city: str) -> str:
        """Get the weather of the given city.

        Args:
            city: name of the city whose weather we need.

        """
        return "Sunny, 24 Degree Celcius."

    return {
        "multiply": multiply,
        "get_weather": get_weather,
    }

# # Let's inspect some of the attributes associated with the tool.
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2,
)
tools = get_tools()
llm_with_tools = llm.bind_tools(list(tools.values()))

messages = []

while True:
    user_msg = input("Please ask any question: ")
    messages.append(HumanMessage(user_msg))
    ai_msg = llm_with_tools.invoke(messages)

    for tool_call in ai_msg.tool_calls: # type: ignore
        try:
            selected_tool = tools[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)
            print(tool_call["name"], tool_call)
        except Exception as e:
            print(e)
            messages.append(ToolMessage("No tool found!"))

    final_answer = llm_with_tools.invoke(messages)

    print("AI:", final_answer.content)
