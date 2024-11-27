"""General LLM Chat Bot."""
from typing import Dict, List, cast

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import jada_smith_msgs


class ConverstationStateManger:
    """Class to interact with session state."""

    __slots__ = "msg_history", "system_msg", "msg_dict_history"

    def __init__(self, system_msg: str="You are a helpful assistant") -> None:
        """Initialize history if not done."""
        if "msg_history" not in st.session_state:
            st.session_state.msg_history = []
            st.session_state.msg_dict_history = []
        # This does nothing it's just for type hints to work properly
        self.msg_history = cast(List[BaseMessage], st.session_state.msg_history)
        self.msg_dict_history = cast(List[Dict[str, str]], st.session_state.msg_dict_history)
        self.system_msg = SystemMessage(system_msg)

    def add_message(self, msg: BaseMessage) -> None:
        """Add messge to history."""
        self.msg_history.append(msg)
        self.msg_dict_history.append(self.msg_obj_to_dict(msg))

    def msg_obj_to_dict(self, msg: BaseMessage) -> Dict[str, str]:
        """Convert msg (AIMessage|HumanMessage) to dict.

        Example: HumanMessage("Hello") -> {"role": "user", "content": "Hello"}
        """
        if isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Invalid msg type: {type(msg)}")
        return { "role": role, "content": str(msg.content) }

    def get_history(self) -> List[BaseMessage]:
        """Return msg history."""
        return self.msg_history

    def get_msg_dict_history(self) -> List[dict]:
        """Return msg history."""
        return self.msg_dict_history

    def get_chat_template(self) -> ChatPromptTemplate:
        """Return chat template."""
        return ChatPromptTemplate([
            self.system_msg,
            *self.get_history()[-50:],
            ("human", "{input}"),
        ])

class UI:
    """Class to handle writing to UI."""

    __slots__ = ("state",)

    def __init__(self, title: str, state: ConverstationStateManger, sub_header: str) -> None:
        """Initialize UI."""
        st.title(title)
        self.state = state

        st.write(sub_header)
        self.write_history()

    def write_history(self) -> None:
        """Write msg history from state to UI."""
        for msg in self.state.get_history():
            self.write_msg(msg)

    def write_msg(self, msg: BaseMessage) -> None:
        """Write single msg to UI."""
        msg_dict = self.state.msg_obj_to_dict(msg)
        with st.chat_message(msg_dict["role"]):
            st.markdown(msg_dict["content"])

    def chat_input(self) -> HumanMessage | None:
        """Chat bot input."""
        content = st.chat_input()
        if not content:
            return None
        human_msg = HumanMessage(content)
        self.write_msg(human_msg)
        return human_msg

def main() -> None:
    """Run the loop of streamlit."""
    state = ConverstationStateManger(system_msg=jada_smith_msgs.system_prompt)
    ui = UI(
        "Interview with Jada Smith",
        state,
        sub_header=jada_smith_msgs.context_for_user
    )
    model = ChatOllama(model="llama3.2",base_url="http://localhost:11434", temperature=0.2)

    if user_msg := ui.chat_input():
        prompt = user_msg.content
        chain = state.get_chat_template() | model | StrOutputParser()
        response = st.write_stream(chain.stream({ "input": prompt, }))
        state.add_message(user_msg)
        state.add_message(AIMessage(response))



if __name__ == "__main__":
    main()
