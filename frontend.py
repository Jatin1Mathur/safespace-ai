#step 1 : setup streamlit
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/ask"

st.set_page_config(page_title = "AI Mental Health Therapist" , layout = "wide")
st.title(" safeSpace AI Mental Health Therapist")

#Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#step 2 : User is able to ask question 
user_input = st.chat_input("what's on your mind today?")
if user_input:
    st.session_state.chat_history.append({"role": "user" , "content" : user_input})
    fixed_dummy_response_from_backend  = requests.post(BACKEND_URL , json = {"message": user_input})

    # ✅ FIX APPLIED HERE
    try:
        data = fixed_dummy_response_from_backend.json()
        content = data.get("response", "No response")
    except Exception:
        content = fixed_dummy_response_from_backend.text

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": content
    })

#step 3 : show response from backend
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
