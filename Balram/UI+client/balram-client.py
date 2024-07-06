import streamlit as st
from langserve import RemoteRunnable

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to interact with the FastAPI chatbot
def chatbot(input_text):
    remote_chain = RemoteRunnable("http://localhost:9000/agent")
    response = remote_chain.invoke({
        'input':str(input_text),
        'chat_history': []
    })
    return response['output']

def add_message_to_history(user_msg, bot_msg):
    st.session_state.chat_history.append((user_msg, bot_msg))

def display_chat_history():
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"<div style='text-align:right; border: 1px solid #ccc; border-radius: 5px; padding: 10px;'>User: {user_msg}</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(f"<div style='text-align:left; border: 1px solid #ccc; border-radius: 5px; padding: 10px;'>Bot: {bot_msg}</div>", unsafe_allow_html=True)
        st.write("")

# Streamlit UI
def main():
    st.title("BALRAM")

    # Input text area for user queries
    user_input = st.chat_input("Enter your message here:")

    if user_input:
        bot_response = chatbot(user_input)
        add_message_to_history(user_input, bot_response)
        
    display_chat_history()

if __name__ == "__main__":
    main()
