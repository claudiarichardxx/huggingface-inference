from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import torch
import asyncio

app = FastAPI(
            title="MBTI Inference API",
            description="Predict MBTI personality types using a fine-tuned BERT model.",
            version="1.0.0"
            )

'''
Load the pre-trained model and tokenizer. This is done at the start to avoid reloading on each request.

Model: ClaudiaRichard/mbti-bert-nli-finetuned_v2

Why this model?

Rather than using a standard sentiment classifier, I wanted to showcase something a bit more personal and intellectually engaging: 
MBTI personality prediction. This model is based on BERT and fine-tuned by me to classify text into MBTI types like INTJ, ENFP, and others. 
I've always been fascinated by how people express their personalities through language. The MBTI framework adds an extra layer of structure 
that makes the task more interesting — it's not just a single-label prediction, but a multi-dimensional classification problem.

Choosing this model reflects my curiosity about both human behavior and machine learning. It's also a great technical showcase for 
multi-label classification, transfer learning with Hugging Face Transformers, and efficient API design for real-time inference.

And of course, it's a lot more fun to say, “Tell me something about yourself and I'll guess your personality,” than 
                                                                                                    “This is positive or negative.”

'''

model_name = "ClaudiaRichard/mbti-bert-nli-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()


class InferenceRequest(BaseModel):
    
    '''Text input for MBTI prediction. Defines the structure of the request body for the prediction endpoint.'''

    text: str


@app.post("/predict")
async def predict(request: InferenceRequest) -> dict: 

    '''
    Endpoint to predict MBTI type based on input text. Adding async support for better performance.

    Input: request (InferenceRequest) - Contains the text to be analyzed.
    Output: dict - A dictionary containing the predicted MBTI type.

    How it works:
    1. The input text is extracted from the request.
    2. The text is passed to a synchronous function that handles the prediction logic.
    3. The synchronous function tokenizes the text, passes it through the model, and returns the predicted MBTI type.
    4. The result is returned as a JSON response.
    '''

    return await asyncio.to_thread(sync_predict, request.text)

def sync_predict(text: str) -> dict:

    '''
    Synchronous function to handle prediction logic.

    Input: text (str) - The input text for MBTI prediction.
    Output: dict - A dictionary containing the predicted MBTI type.

    How it works:
    1. Tokenizes the input text using the pre-trained tokenizer.
    2. Passes the tokenized input through the model to get logits.
    3. Applies sigmoid activation to convert logits to probabilities.
    4. Sets a threshold of 0.6 to determine class predictions.
    5. Maps the predicted classes to MBTI types.
    6. Returns a dictionary with the predicted MBTI labels.
    '''

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate probabilities for each class
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())

        predictions = np.zeros(probs.shape)

        predictions[np.where(probs >= 0.6)] = 1
        predicted_labels =  tuple(predictions)

        mbti_mapping = {0: ['Introvert', 'Extrovert'], 1: ['Intuition', 'Sensing'], 2: ['Thinking', 'Feeling'], 3: ['Judging', 'Perceiving']}
        mbti_letter_mapping = {0: ['I', 'E'], 1: ['N', 'S'], 2: ['T', 'F'], 3: ['J', 'P']}
        predicted_mbti = []
        predicted_letters = ''
        # Map predicted labels to MBTI types
        for i in range(4):
            predicted_mbti.append(mbti_mapping[i][int(predicted_labels[i])])
            predicted_letters += mbti_letter_mapping[i][int(predicted_labels[i])]

    return {
            "labels": predicted_mbti, 
            "mbti": predicted_letters
            }

