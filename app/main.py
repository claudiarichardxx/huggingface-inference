from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
from utils.predict import Predictions
from utils.attributions import Attributions
from utils.train import FineTuner

import os
from dotenv import load_dotenv
load_dotenv()

from logger import logger


app = FastAPI(
            title="MBTI Inference API",
            description="Predict MBTI personality types using a fine-tuned BERT model.",
            version="1.0.0"
            )

'''
My finetuned model: ClaudiaRichard/mbti-bert-nli-finetuned_v2


Rather than using a standard sentiment classifier, I wanted to showcase something a bit more personal and intellectually engaging: 
MBTI personality prediction. This model is based on BERT and fine-tuned by me to classify text into MBTI types like INTJ, ENFP, and others. 
I've always been fascinated by how people express their personalities through language. The MBTI framework adds an extra layer of structure 
that makes the task more interesting — it's not just a single-label prediction, but a multi-dimensional classification problem.

Choosing this model reflects my curiosity about both human behavior and machine learning. It's also a great technical showcase for 
multi-label classification, transfer learning with Hugging Face Transformers, and efficient API design for real-time inference.

And of course, it's a lot more fun to say, “Tell me something about yourself and I'll guess your personality,” than 
                                                                                                    “This is positive or negative.”

'''
default_model_name = os.getenv("DEFAULT_MODEL_NAME", "ClaudiaRichard/mbti-bert-nli-finetuned")


class InferenceRequest(BaseModel):
    
    '''Text input for MBTI prediction. Defines the structure of the request body for the prediction endpoint.'''
    
    text: str
    model_name: Optional[str] = default_model_name


@app.get("/predict")
def process_inference_requests(request: InferenceRequest) -> dict: 

    '''
    Endpoint to predict MBTI type based on input text. Adding async support for better performance.

    Args: 
        request (InferenceRequest) - Contains the text to be analyzed.
    Returns: 
        dict - Predicted MBTI type and attribution values.

    example request body:
    {
        "text": "i like parties because ..."
    }

    example response body:
    {
    "labels": [
        "Introvert",
        "Intuition",
        "Thinking",
        "Perceiving"
    ],
    "mbti": "INTP",
    "refined_labels": [
        "Inconclusive",
        "Inconclusive",
        "Thinking",
        "Perceiving"
    ],
    "attribution_values": {
        "Introvert_Extrovert": {
            "[CLS]": 0.308,
            "i": 1.0,
            "like": 0.7747,
            "parties": 0.303,
            "because": 0.4498,
            ".": 0.43,
            "[SEP]": 0.72
        },
        "Intuition_Sensing": {
            "[CLS]": 0.3033,
            "i": 0.5843,
            "like": 1.0,
            "parties": -0.2091,
            "because": -0.0692,
            ".": 0.4577,
            "[SEP]": -0.2636
        },
        "Thinking_Feeling": {
            "[CLS]": -0.1523,
            "i": -0.6047,
            "like": 0.1917,
            "parties": -1.0,
            "because": -0.2585,
            ".": -0.3671,
            "[SEP]": -0.9737
        },
        "Judging_Perceiving": {
            "[CLS]": 0.0665,
            "i": 1.0,
            "like": 0.2635,
            "parties": 0.2664,
            "because": -0.1741,
            ".": -0.4705,
            "[SEP]": -0.1914
        }
    }
}

    what it does:
    1. Loads the specified model (or default if none specified).
    2. Tokenizes the input text.
    3. Predicts MBTI type using the model.
    4. Computes token-level attributions for interpretability.
    5. Applies thresholding to refine predictions.
    6. Returns the predicted MBTI type along with attribution values for each token.
    '''
    
    logger.info("Received inference request.")
    logger.info(f"Using model: {request.model_name}")
    pred = Predictions(model_name=default_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred.model.to(device) 
    predictions = pred.predict_labels(request.text)
    result = pred.map_and_return(predictions)

    logger.info("Predicted MBTI for input text, trying to compute attributions and refine labels.")
    encoded = pred.tokenizer(request.text, return_attention_mask=True, return_tensors="pt")
    tokens = pred.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    logger.info("Computing attributions for interpretability.")
    attribute = Attributions(pred.model)


    logger.info("Refining labels based on attribution thresholds.")
    labs = {0: ['Introvert', 'Extrovert'], 1: ['Intuition', 'Sensing'], 2: ['Thinking', 'Feeling'], 3: ['Judging', 'Perceiving']}
    refined_labels = []
    attribution_values = {}

    logger.info("Storing token-level attributions.")
    for i in range(4):
        attributions = attribute.getAttributions(encoded, i)
        label_index = int(predictions[i])

        # Apply thresholding logic
        if checkThreshold(attributions, label=label_index):
            refined_labels.append(labs[i][label_index])
        else:
            refined_labels.append("Inconclusive")

        attributions = attributions.squeeze()  

        # Store token-level attribution values
        attribution_values[labs[i][0] + "_" + labs[i][1]] = {
                token: round(attributions[j].item(), 4) for j, token in enumerate(tokens)
            }

    result['refined_labels'] = refined_labels
    result['attribution_values'] = attribution_values
    logger.info("Attributions computed and labels refined.")

    return result

def checkThreshold(attributions, label, threshold = [-0.3, 0.6]):
        
        '''
        Check if attributions meet the threshold criteria for refining labels.
        Args: 
            attributions: np.array - Attribution values for each token.
            label: int - Predicted label index (0 or 1).
            threshold: list - Threshold values for determining label confidence.
        Returns: 
            bool - True if threshold criteria are met, False otherwise.

        Logic:
            For label 0 (e.g., Introvert), check if any attribution is less than or equal to -0.3.
            For label 1 (e.g., Extrovert), check if any attribution is greater than or equal to 0.6.
        '''

        if(label == 0):
            if(any(attributions[0]<= threshold[0])):
                return True
            else:
                return False
            
        if(label == 1):
            if(any(attributions[0] >= threshold[1])):
                return True
            else:
                return False


class TrainRequest(BaseModel):
    
    '''Text input for MBTI prediction. Defines the structure of the request body for the prediction endpoint.'''

    base_model: str
    dataset_name: str
    finetuned_model_name: str


@app.post("/train")
def train_model(request: TrainRequest):

    # print("Starting training with base model:", request.base_model)
    '''
    Endpoint to fine-tune a model on MBTI classification task.
    Args:
        request (TrainRequest) - Contains base model name, dataset name, and new model name.
    Returns:
        dict - Training history and evaluation results.
        
    sample request body:
    {
        "base_model": "bert-base-uncased",
        "dataset_name": "ClaudiaRichard/mbti_classification_v2",
        "finetuned_model_name": "mbti-bert-finetuned-test"
    }
    
    sample response body:
    {
        "history": {
            "train_runtime": 1234.56,
            "train_samples_per_second": 8.1,
            "train_steps_per_second": 0.5,
            "total_flos": 1.234e+16,
            "train_loss": 0.5678,
            "epoch": 1.0
        },
        "test_results": {
            "eval_loss": 0.4567,
            "eval_f1": 0.789,
            "eval_roc_auc": 0.8765,
            "eval_runtime": 45.67,
            "eval_samples_per_second": 22.0,
            "eval_steps_per_second": 1.2
        }
    }
    
    what it does:
    1. Loads the specified dataset and base model.
    2. Preprocesses the dataset for MBTI classification.
    3. Sets up training arguments and trainer.
    4. Fine-tunes the model on the training set.
    5. Evaluates the model on the test set.
    6. Returns training history and evaluation metrics.
    '''

    logger.info("Received training request.")
    ft = FineTuner(dataset_name = request.dataset_name, base_model = request.base_model)
    logger.info(f"Starting training with base model: {request.base_model}, dataset: {request.dataset_name}, new model name: {request.finetuned_model_name}")
    ft_results = ft.train(new_model_name = request.finetuned_model_name)


    return ft_results


# if __name__ == "__main__":

#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)




