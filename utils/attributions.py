from captum.attr import LayerIntegratedGradients
import torch
from typing import Tuple, Dict, Any
import numpy as np

from logger import logger

class Attributions:

    '''
    Class to compute attributions using Layer Integrated Gradients.'''

    def __init__(self, model: torch.nn.Module) -> None:

        '''
        Initialize the Attributions class with the model and Layer Integrated Gradients.
        Args:
            model: torch.nn.Module - Pretrained model for which attributions are to be computed.

        Returns:
            None

        Logic:
            1. Store the model.
            2. Initialize Layer Integrated Gradients with the model's embedding layer.
        '''

        self.model = model
        self.lig = LayerIntegratedGradients(self.calculate, model.bert.embeddings)
        logger.info("Attributions class initialized with model and Layer Integrated Gradients.")

    def calculate(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        '''
        Forward function for the model.
        Args:
            input_ids: torch.Tensor - Token IDs.
            token_type_ids: torch.Tensor - Token type IDs.
            attention_mask: torch.Tensor - Attention mask.
            
        Returns:
            torch.Tensor - Model outputs (logits).
            
        Logic:
            1. Pass the inputs through the model.
            2. Return the logits for further processing.
        '''
    
        return self.model(input_ids, token_type_ids, attention_mask)[0]
    

    def getAttributions(self, encoded: Dict[str,torch. Tensor], target: int) -> np.array:

        '''
        Compute attributions for the given input and target class.
        Args:
            encoded: dict - Tokenized input containing input_ids, token_type_ids, and attention_mask.
            target: int - Target class index for which attributions are to be computed.
            
        Returns:
            np.array - Normalized attributions for each token.
            
        Logic:
            1. Create a baseline tensor of zeros with the same shape as input_ids.
            2. Use Layer Integrated Gradients to compute attributions.
            3. Sum attributions across the embedding dimension.
            4. Normalize attributions by dividing by the maximum absolute value.
            5. Return the normalized attributions.
        '''

        bsl = torch.zeros(encoded.input_ids.size()).type(torch.LongTensor)
        attributions, delta = self.lig.attribute(
                inputs = encoded["input_ids"],
                baselines = bsl,
                additional_forward_args = (
                    encoded["token_type_ids"],
                    encoded["attention_mask"]
                ),
                n_steps = 20,
                target = target,
                return_convergence_delta = True,
            )
        attributions_ig = attributions.sum(dim=-1).cpu()
        attributions_ig = attributions_ig / attributions_ig.abs().max()

        return attributions_ig
        

