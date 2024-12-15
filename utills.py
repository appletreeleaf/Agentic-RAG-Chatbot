# Streamlit
import streamlit as st

# Logging library
from loguru import logger

# LangChain Core messages
from langchain_core.messages import ChatMessage

# LangChain Core callbacks
from langchain_core.callbacks.base import BaseCallbackHandler

# Chat message history
from langchain_community.chat_message_histories import ChatMessageHistory

# Base chat message history
from langchain_core.chat_history import BaseChatMessageHistory


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def print_message():
    if "message" in st.session_state and st.session_state["message"]:
        for chat_message in st.session_state["message"]:
            st.chat_message(chat_message.role).write(chat_message.content)


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환
