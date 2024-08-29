import os
import streamlit as st
import requests

st.title("RAG")
st.write("Upload PDFs and Chat.")

# for DOCKER
#BACKEND_API = os.getenv('BACKEND_API', 'http://api:8000')

# for Kubernetes
BACKEND_API = os.getenv('BACKEND_API', 'http://my-api-service:8000')

if 'history' not in st.session_state:
    st.session_state.history = []

if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

def send_message():
    user_input = st.session_state.input
    if user_input:
        try:
            response = requests.post(f"{BACKEND_API}/chat", json={"message": user_input}, timeout=None)
            #response = requests.post(f"http://{BACKEND_API}:8000/chat", json={"message": user_input}, timeout=None)
            response_data = response.json()

            bot_response = response_data.get("response", "No response from the bot")

            st.session_state.history.append({"sender": "You", "message": user_input})
            st.session_state.history.append({"sender": "AI", "message": bot_response})

        except requests.exceptions.RequestException as e:
            bot_response = f"Error: {e}"
            st.session_state.history.append({"sender": "You", "message": user_input})
            st.session_state.history.append({"sender": "AI", "message": bot_response})

        st.session_state.input = ""


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# BACKEND_API = "rag_chatbot-api-1"
if uploaded_file and not st.session_state.file_processed:
    with st.spinner("Processing the PDF..."):
        response = requests.post(f"{BACKEND_API}/upload", files={"file": uploaded_file})
        # response = requests.post(f"http://{BACKEND_API}:8000/upload", files={"file": uploaded_file})
        if response.status_code == 200:
            st.success("PDF processed successfully!")
            response_data = response.json()
            text_file_path = response_data.get("text_file_path", None)
            if text_file_path:
                st.write(f"Extracted text saved at: {text_file_path}")
            st.session_state.file_processed = True 
        else:
            st.error("Failed to process the PDF.")

user_input = st.text_input("You:", key="input", on_change=send_message)

for chat in st.session_state.history:
    if chat["sender"] == "You":
        st.markdown(f"<div style='text-align: right; margin: 10px 0;'><b>{chat['sender']}:</b></div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color: #e0f7fa; padding: 10px; border-radius: 5px; margin: 10px 0; text-align: right;'>{chat['message']}</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; margin: 10px 0;'><b>{chat['sender']}:</b></div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color: #ffecb3; padding: 10px; border-radius: 5px; margin: 10px 0;'>{chat['message']}</div>",
            unsafe_allow_html=True)
