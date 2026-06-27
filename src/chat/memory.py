import streamlit as st


def initialize_chat():
    """
    Initialize chat history.
    """

    if "messages" not in st.session_state:

        st.session_state.messages = []


def add_user_message(message):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": message,
        }
    )


def add_ai_message(message):

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": message,
        }
    )


def display_chat():

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])
def get_chat_history():

    history = ""

    for message in st.session_state.messages:

        role = message["role"].capitalize()

        history += f"{role}: {message['content']}\n"

    return history