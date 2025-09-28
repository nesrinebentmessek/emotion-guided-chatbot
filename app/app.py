import streamlit as st
from src.model_loader import load_models
from src.chatbot_utils import detect_emotion, generate_response

st.set_page_config(page_title="Emotion-Aware Chatbot", layout="centered")

st.title("Emotion-Aware Chatbot")
st.markdown("Type something and the chatbot will respond with awareness of your **emotion**.")

# Load models (cached)
@st.cache_resource
def load_all_models():
    return load_models()

chat_tokenizer, chat_model, emo_tokenizer, emo_model, emotion_dict = load_all_models()

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.text_input("You:", "")

if user_input:
    # Detect emotion
    emotion = detect_emotion(user_input, emo_model, emo_tokenizer, emotion_dict)
    # Generate chatbot response
    bot_response = generate_response(user_input, emotion, chat_model, chat_tokenizer)

    # Save history
    st.session_state.history.append({"user": user_input, "bot": bot_response, "emotion": emotion})

# Show chat history
for chat in st.session_state.history[::-1]:
    st.markdown(f"🧑 **You:** {chat['user']}")
    st.markdown(f"🎭 **Emotion detected:** *{chat['emotion']}*")
    st.markdown(f"🤖 **Bot:** {chat['bot']}")
    st.markdown("---")
 
