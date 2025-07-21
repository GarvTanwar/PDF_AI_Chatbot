import streamlit as st
from utils.api import ask_question
import os

def render_chat():
    st.subheader("💬 Chat with the Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input prompt
    user_input = st.chat_input("Ask your Question...")

    if user_input:
        # Display and store user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Show assistant thinking spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_question(user_input)

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("response", "🤔 No answer received.")
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            st.markdown("**📚 Sources:**")
                            unique_sources = sorted(set(sources))
                            for src in unique_sources:
                                filename = os.path.basename(src)
                                st.markdown(f"- `{filename}`")

                        # Append assistant reply to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                    else:
                        st.error(f"❌ Error: {response.status_code} — {response.text}")
                except Exception as e:
                    st.error(f"⚠️ An exception occurred: {e}")
