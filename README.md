# Emotion-Aware Chatbot

This repository contains a project to build an emotion-aware chatbot using fine-tuned transformer models. The chatbot detects user emotions with a fine-tuned DistilBERT model and generates empathetic responses using a fine-tuned DialoGPT model. The models are trained on publicly available datasets for emotion classification and conversational responses.

The project is implemented in three Jupyter notebooks, designed to run in Google Colab (with GPU acceleration recommended for training).

## Project Overview

- **Emotion Detection**: Fine-tune DistilBERT on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset to classify text into 6 emotions (sadness, joy, love, anger, fear, surprise).
- **Chatbot Response Generation**: Fine-tune DialoGPT on the [DailyDialog](https://huggingface.co/datasets/daily_dialog) dataset, incorporating emotion tags to make responses contextually aware.
- **Testing & Inference**: Evaluate the models and run an interactive chat demo where the chatbot responds based on detected emotions.

Key insights:
- DistilBERT achieves high accuracy (~93% on test set) for emotion classification.
- DialoGPT shows a validation loss of ~2.03 after 10 epochs, generating coherent emotion-sensitive responses.
- Test perplexity for DialoGPT is ~277, indicating room for improvement with more training or larger models.

## Notebooks

1. **01_Fine-Tuning_DistilBERT.ipynb**: Loads the emotion dataset, preprocesses data, fine-tunes DistilBERT, evaluates on test set (accuracy, F1-score), and saves the model. Includes an interactive prediction demo.
   
2. **02_Fine-Tuning_DialoGPT.ipynb**: Loads the DailyDialog dataset, adds emotion tags to dialogues, fine-tunes DialoGPT-small, evaluates on validation set, and saves the model.

3. **03_Testing_and_inference.ipynb**: Loads the fine-tuned models, evaluates DialoGPT on the test set (loss and perplexity), and runs an interactive chat session where emotions are detected and responses are generated.

## Requirements

The notebooks use the following libraries (install via pip in Colab):

```python
!pip install torch torchvision torchaudio
!pip install transformers accelerate evaluate
!pip install datasets==3.6.0
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

Models are saved to `/content/drive/MyDrive/DialoGPT-emotion` (update paths as needed).

## Results

- **DistilBERT (Emotion Classifier)**:
  - Test Accuracy: ~93%
  - F1-Score: ~93% (macro average)
  - Confusion matrix shows strong performance across emotions.

- **DialoGPT (Chatbot)**:
  - Validation Loss: 2.031
  - Test Loss: 5.626 | Perplexity: 277.44
  - Generates contextually appropriate responses, though fluency can vary.

Future improvements: Use larger models (e.g., DialoGPT-medium), more epochs.

## Credits

- Datasets: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) and [DailyDialog](https://huggingface.co/datasets/daily_dialog).
- Models: Hugging Face's DistilBERT and DialoGPT.
- Inspired by conversational AI research for emotional intelligence.


