from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Tuple, Dict, Any
import logging


from logger import logger



class Predictions:

    def __init__(self, model_name: str) -> None:

        '''
        Load the pre-trained model and tokenizer.
        Args:
            model_name: str - Name or path of the pre-trained model.
        
        Returns:
            None - Loads the model and tokenizer into the class instance.
        
        Logic:
            1. Attempt to load the model and tokenizer from local directories.
            2. If not found, load from the specified model name/path and save locally for future use.
        '''

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(f"models/{model_name}/model/")
            self.tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}/tokenizer/")
            logger.info("Model and tokenizer loaded from local directory.")

        except:
            logger.info("Local model not found. Loading from specified model name/path.")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.save_pretrained(f"models/{model_name}/model/")
            self.tokenizer.save_pretrained(f"models/{model_name}/tokenizer/")


    def predict_labels(self, text: str) -> Tuple[int, int, int, int]:

        '''
        Predict MBTI labels for the given text.
        Args:
            text: str - Input text for prediction.
        
        Returns:
            tuple - Predicted labels for each MBTI dimension.
            
        Logic:
            1. Tokenize the input text.
            2. Pass the tokenized input through the model to get logits.
            3. Apply a sigmoid function to convert logits to probabilities.
            4. Use a threshold to determine binary labels for each MBTI dimension.
        '''

        encoding = self.tokenizer(text, return_attention_mask=True, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        outputs = self.model(**encoding)
        logits = outputs.logits  # Shape: (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.6)] = 1
        logger.info("Predictions computed.")

        return tuple(predictions)
    
    
    def map_and_return(self, predicted_labels: Tuple[int, int, int, int]) -> Dict[str, Any]:

        '''
        Map predicted binary labels to MBTI types.
        Args:
            predicted_labels: tuple - Binary labels for each MBTI dimension.
            
        Returns:
            dict - Mapped MBTI types and their corresponding letters.
            
        Logic:
            1. Define mappings for each MBTI dimension.
            2. Iterate through the predicted labels and map them to their respective MBTI types.
            3. Concatenate the letters to form the full MBTI type.
            4. Return a dictionary with both the full types and the concatenated letters.
        '''

        mbti_mapping = {0: ['Introvert', 'Extrovert'], 1: ['Intuition', 'Sensing'], 2: ['Thinking', 'Feeling'], 3: ['Judging', 'Perceiving']}
        mbti_letter_mapping = {0: ['I', 'E'], 1: ['N', 'S'], 2: ['T', 'F'], 3: ['J', 'P']}


        predicted_mbti = []
        predicted_letters = ''
        for i in range(4):
            predicted_mbti.append(mbti_mapping[i][int(predicted_labels[i])])
            predicted_letters += mbti_letter_mapping[i][int(predicted_labels[i])]

        logger.info("Predicted MBTI types mapped.")

        return {
                "labels": predicted_mbti, 
                "mbti": predicted_letters
                }