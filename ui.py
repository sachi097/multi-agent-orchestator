import streamlit as st
import requests

# Streamlit App Configuration
st.set_page_config(page_title="Multi Agent Orchestrator UI", layout="centered")

# Define API endpoint
API_URL = "http://127.0.0.1:8080/orchestrated_chat"


# Input box for user messages
user_id = st.text_area("Enter your user id:", height=68, placeholder="Type your user id here...")
user_input = st.text_area("Enter your message(s):", height=150, placeholder="Type your message here...")

# Button to send the query
if st.button("Submit"):
    if user_input.strip():
        try:
            
            # Send the input to the FastAPI backend
            payload = {"user_input": user_input, "user_id": user_id, "session_id": "123456"}
            response = requests.post(API_URL, json=payload, stream=True)
            
            # Display the response
            if response.status_code == 200:
                st.subheader("Agent Response:")
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        st.markdown(f"{decoded_line}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message before clicking 'Send Query'.")