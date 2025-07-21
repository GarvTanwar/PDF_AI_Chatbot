import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chat_ui import render_chat

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("AI PDF Chatbot")

render_uploader()
render_chat()
render_history_download()