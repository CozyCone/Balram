import streamlit as st
from langserve import RemoteRunnable

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to interact with the FastAPI chatbot
def chatbot(input_text):
    remote_chain = RemoteRunnable("http://localhost:9000/agent")
    response = remote_chain.invoke({
        'input': str(input_text),
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
    st.title("BALRAM - Farmer Assist")

    # Render the header section
    st.markdown("""
    <header style="text-align: center; background-color: #4CAF50; color: white; padding: 1.5rem;">
      <h1 style="font-size: 2rem;">Farmer Assist</h1>
      <p>Your guide for stable farming income</p>
    </header>
    """, unsafe_allow_html=True)

    # Display buttons for quick actions
    st.write("## What would you like to explore?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button('Look for Government Schemes', on_click=lambda: st.write("[Government Schemes](https://agriwelfare.gov.in/en/Major)"))
    
    with col2:
        st.button('Find Crop Prices', on_click=lambda: st.write("[Crop Prices](https://agmarknet.gov.in/PriceTrends/)"))
    
    with col3:
        st.button('Get Help with Farming Equipment')

    st.write("## Trending Questions")
    st.markdown("""
    - How to access subsidies for drip irrigation?
    - Where to sell organic crops directly?
    - Best crops to grow during monsoon?
    - What are the current prices for wheat?
    - How to start millet farming?
    """)
    
    st.write("## Ask Your Queries")

    # Input text area for user queries
    user_input = st.chat_input("Enter your message here:")

    if user_input:
        bot_response = chatbot(user_input)
        add_message_to_history(user_input, bot_response)
        
    display_chat_history()

if __name__ == "__main__":
    main()
