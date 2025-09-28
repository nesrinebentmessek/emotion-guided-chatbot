import streamlit as st
from src.model_loader import load_models
from src.chatbot_utils import detect_emotion, generate_response

st.set_page_config(page_title="Friendly Emotion-Aware Chatbot", layout="centered")

st.title("Friendly Emotion-Aware Chatbot")
st.markdown("Type something .")

@st.cache_resource
def load_all_models():
    return load_models()


chat_tokenizer, chat_model, emo_tokenizer, emo_model, emotion_dict = load_all_models()

if "history" not in st.session_state:
    st.session_state.history = []


user_input = st.text_input("You:", "")
if user_input:
    emotion = detect_emotion(user_input, emo_model, emo_tokenizer, emotion_dict)
    bot_response = generate_response(user_input, emotion, chat_model, chat_tokenizer)
    st.session_state.history.append({"user": user_input, "bot": bot_response, "emotion": emotion})


for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Emotion detected:** *{chat['emotion']}*")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
 
