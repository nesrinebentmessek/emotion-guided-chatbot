import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(dialo_dir="models/DialoGPT-emotion", emotion_dir="models/distilbert-emotion"):
    # Load DialoGPT
    chat_tokenizer = AutoTokenizer.from_pretrained(dialo_dir)
    chat_model = AutoModelForCausalLM.from_pretrained(dialo_dir).to(device)

    # Load DistilBERT emotion model
    emo_tokenizer = AutoTokenizer.from_pretrained(emotion_dir)
    emo_model = DistilBertForSequenceClassification.from_pretrained(emotion_dir).to(device)

    # Emotion labels
    emotion_dict = {
        0: 'neutral', 1: 'anger', 2: 'disgust', 
        3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'
    }

    return chat_tokenizer, chat_model, emo_tokenizer, emo_model, emotion_dict
 
