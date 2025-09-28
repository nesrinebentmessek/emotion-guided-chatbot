import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_emotion(user_input, emo_model, emo_tokenizer, emotion_dict):
    inputs = emo_tokenizer(user_input, return_tensors="pt").to(device)
    outputs = emo_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    emotion_id = torch.argmax(probs, dim=-1).item()
    return emotion_dict[emotion_id]

def generate_response(user_input, detected_emotion, chat_model, chat_tokenizer, max_length=100):
    prompt = f"{detected_emotion}: {user_input}"
    input_ids = chat_tokenizer.encode(prompt + chat_tokenizer.eos_token, return_tensors="pt").to(device)
    response_ids = chat_model.generate(input_ids, max_length=max_length, pad_token_id=chat_tokenizer.eos_token_id)
    response = chat_tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response
 
