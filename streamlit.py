import os
import re
import streamlit as st
# ... (other imports)

# ... (CustomChatbot class - as provided in my previous corrected answer)

def clean_response_string(text):
    # ... (no changes)

def extract_text(data, path=""):
    # ... (no changes)

def generate_response(input_text):
    try:
        bot = get_chatbot()  # Get the chatbot instance
        if bot is None:  # Check if chatbot initialization was successful
            st.error("Chatbot initialization failed. Please check the logs.")
            return "A problem occurred during chatbot initialization." # Return a string to avoid further errors
        response = bot.ask(input_text)

        # ... (rest of the generate_response function - no changes)
        return "\n\n".join(f"- {part}" for part in formatted_response)

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return f"Sorry, there was an error processing your request: {e}"

@st.cache_resource
def get_chatbot(pdf_path='gpmc.pdf'):
    try:
        return CustomChatbot(pdf_path=pdf_path)
    except Exception as e:
        st.error(f"Error creating chatbot: {e}")
        return None  # VERY IMPORTANT: Return None if chatbot creation fails

# Streamlit app
# ... (rest of your Streamlit code)

if input_text := st.chat_input("Type your question here..."):
    # ...
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(input_text)  # Call generate_response
            if isinstance(response, str) and len(response) > 100:
                st.markdown(response)
            else:
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
