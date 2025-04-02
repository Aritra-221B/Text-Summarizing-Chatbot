import streamlit as st
import re
from transformers import pipeline

# Load the improved model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="t5-base")

summarizer = load_model()

# Function to fix capitalization in AI-generated summaries
def fix_capitalization(text):
    """Ensure proper sentence capitalization in AI-generated summaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence endings
    corrected_sentences = [s.capitalize() for s in sentences]  # Capitalize first letter of each sentence
    return " ".join(corrected_sentences)  # Rejoin sentences properly

# Streamlit UI Configuration
st.set_page_config(page_title="Summarization Chatbot", page_icon="👾", layout="wide")

# Sidebar
st.sidebar.title("💡 **About My Chatbot**")
st.sidebar.info(
    """
    Welcome to the **AI-Powered Summarization Chatbot!** 🤖  
    This tool helps you quickly summarize and understand large pieces of text by generating **concise, well-structured, and paraphrased summaries**.  

    ### 📌 How to Use:
    1. Enter or paste a paragraph in the chat input.  
    2. Click **Enter** and let the AI summarize it.  
    3. Get a **clear, concise, and well-structured** summary instantly.  

    🚀 Built with **Hugging Face Transformers** & **Streamlit** for an intuitive experience.  
    ### **Thanks for using my Chatbot!**
    """
)

# Chatbot Title
st.markdown(
    "<h1 style='text-align: center; color: #BB9CD5;'>🤖 Text Summarization Chatbot 📑</h1>", 
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False  # Flag to track if greeting was displayed

# Greeting Message Logic - Avoid Duplication
greeting_message = (
    "👋 Hello! Welcome to the **AI Summarization Chatbot** 🤖.\n\n"
    "I can generate **clear and concise summaries** from any text you provide. "
    "Just enter a paragraph, and I'll create a well-structured summary for you! 🚀\n\n"
    "Go ahead and type or paste some text to get started! ⬇️"
)

# **Ensure Greeting Message is NOT Added Again**
if not st.session_state.greeting_shown:
    if not any(msg["content"] == greeting_message for msg in st.session_state.messages):
        st.session_state.messages.insert(0, {"role": "assistant", "content": greeting_message})
    st.session_state.greeting_shown = True  # Set flag to True

# Display previous messages
for message in st.session_state.messages:
    role = "🧑‍💻 " if message["role"] == "user" else "🤖 "
    with st.chat_message(message["role"]):
        st.markdown(f"**{role}:** {message['content']}")

# User input form
user_input = st.chat_input("Enter your text to summarize...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user input
    with st.chat_message("user"):
        st.markdown(f"**🧑‍💻:** {user_input}")

    # Generate Summary
    with st.spinner("Summarizing your text... ⏳"):
        summary = summarizer(
            user_input, 
            max_length=120, min_length=50, 
            do_sample=True, top_k=50, top_p=0.95, 
            temperature=0.9, num_beams=4, repetition_penalty=1.2, length_penalty=1.0
        )
        summary_text = fix_capitalization(summary[0]["summary_text"])  # Apply capitalization fix

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": summary_text})

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(f"**🤖:** {summary_text}")

# Footer
st.markdown(
    "<div style='text-align: center; margin-top: 50px;'>"
    "Built with ❤️ using Streamlit & Hugging Face Transformers"
    "</div>", 
    unsafe_allow_html=True
)
