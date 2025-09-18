from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score
from datasets import load_dataset
from huggingface_hub import login
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import os
from dotenv import load_dotenv
load_dotenv()

from logger import logger

class MBTITrainer(Trainer):

    '''
    Custom Trainer to handle multi-label classification with class weights.'''

    def __init__(self, *args, class_weights = None, **kwargs):
        '''
        Initialize the MBTITrainer with optional class weights for handling class imbalance.
        Args:
            class_weights: torch.Tensor or None - Weights for each class to handle imbalance.
        Returns:
            None - Initializes the Trainer with custom loss function if class weights are provided.
        '''
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)

        self.loss_fct = torch.nn.BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        Args:
            model: PreTrainedModel - The model to compute the loss for.
            inputs: dict - The inputs and targets of the model.
            return_outputs: bool - Whether to return the model outputs along with the loss.

        Returns:
            loss: torch.Tensor - The computed loss.
            (optional) outputs: ModelOutput - The model outputs if return_outputs is True.

        Logic:
            1. Extract labels from inputs.
            2. Perform a forward pass through the model to get outputs.
            3. Compute the loss using BCEWithLogitsLoss, suitable for multi-label classification.
            4. Return the loss and optionally the outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        try:
            logger.info("Computing loss with model.num_labels")
            loss = self.loss_fct(outputs.logits.view(-1, model.num_labels), labels.view(-1, model.num_labels).float())
        
        except AttributeError:
            logger.info("Computing loss with model.module.num_labels")
            loss = self.loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    

class FineTuner:

    def __init__(self, dataset_name: str, base_model: str) -> None:

        '''
        Load the dataset and the pre-trained model/tokenizer.
        Args:
            dataset_name: str - Name or path of the dataset to load.
            base_model: str - Name or path of the pre-trained model.
            
        Returns:
            None - Loads the dataset, model, and tokenizer into the class instance.
            
        Logic:
            1. Load the dataset from Hugging Face Hub.
            2. Load the pre-trained model and tokenizer, saving them locally for future use.
        '''
        token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=token)
        self.dataset = load_dataset(dataset_name)
        self.labels = ['I/E', 'N/S', 'T/F', 'J/P']

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].select(range(min(10, len(self.dataset[split]))))


        id2label = {0: "I/E", 1: "N/S", 2: "T/F", 3: "J/P"}
        label2id = {"I/E": 0, "N/S": 1, "T/F": 2, "J/P": 3}

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(self.labels),
                                                           id2label=id2label,
                                                           label2id=label2id
                                                           )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("Dataset, model, and tokenizer loaded.")


    def preprocess_data(self, examples: Dict[str, Any]) -> Dict[str, Any]:

        '''
        Preprocess the dataset examples by tokenizing the text and aligning labels.
        Args:
            examples: dict - A batch of examples from the dataset.
            
        Returns:
            dict - Tokenized inputs with aligned labels.
            
        Logic:
            1. Tokenize the text data with padding and truncation.
            2. Extract and align the labels for multi-label classification.
            3. Return the tokenized inputs along with the labels.
        '''


        text = examples["post"]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)

        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        labels_matrix = np.zeros((len(text), len(self.labels)))

        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        logger.info("Data preprocessed and tokenized.")

        return encoding


    def train(self, new_model_name: str) -> Dict[str, Any]:

        '''
        Fine-tune the pre-trained model on the MBTI classification task.
        Args:
            dataset_name: str - Name or path of the dataset to use for training.
            base_model: str - Name or path of the pre-trained model to fine-tune.
            new_model_name: str - Name for the newly fine-tuned model.
            
        Returns:
            dict - Training history and evaluation results.
            
        Logic:
            1. Load the dataset and pre-trained model/tokenizer.
            2. Preprocess the dataset by tokenizing and aligning labels.
            3. Set up training arguments and the custom Trainer.
            4. Fine-tune the model on the training set and evaluate on the test set
            5. Push the fine-tuned model to Hugging Face Hub.
            6. Return the training history and evaluation metrics.
        '''
        
        
        encoded_dataset = self.dataset.map(self.preprocess_data, batched=True, remove_columns=self.dataset['train'].column_names)
        batch_size = 8
        metric_name = "f1"

        args = TrainingArguments(
            'ft/'+new_model_name,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs = 1,
            max_steps=20,           
            fp16=False,   
            weight_decay=0.01,
            gradient_accumulation_steps=2,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
        )

        trainer = MBTITrainer(
                                self.model,
                                args,
                                train_dataset=encoded_dataset["train"],
                                eval_dataset=encoded_dataset["validation"],
                                tokenizer=self.tokenizer,
                                compute_metrics=self.compute_metrics
                            )
        
        torch.cuda.empty_cache()

        logger.info("Starting training.")
        history = trainer.train()
        history_dict = history.metrics if hasattr(history, "metrics") else {}
        results = trainer.evaluate(eval_dataset=encoded_dataset['test'])
        logger.info("Training completed and model evaluated.")
        trainer.push_to_hub(new_model_name)   
        logger.info(f"Model pushed to Hugging Face Hub with name: {new_model_name}")

        return {'history': history_dict, 'test_results': results}
         
        
    def multi_label_metrics(self, predictions: np.array, labels: np.array, threshold: Optional[int] = 0.3):

        '''
        Compute multi-label classification metrics: F1-score and ROC-AUC.
        Args:
            predictions: np.ndarray - Model predictions (logits).
            labels: np.ndarray - True binary labels.
            threshold: float - Threshold to convert logits to binary predictions.
            
        Returns:
            dict - Computed metrics (F1-score and ROC-AUC).
            
        Logic:
            1. Apply a threshold to convert logits to binary predictions.
            2. Compute F1-score using micro averaging.
            3. Compute ROC-AUC score using micro averaging.
            4. Return the metrics as a dictionary.
        '''

        y_pred = (predictions >= threshold).astype(np.float32)

        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')

        metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc}

        return metrics

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, Any]: 

        '''
        Compute metrics for evaluation during training.
        Args:
            p: EvalPrediction - Contains predictions and true labels.
            
        Returns:
            dict - Computed metrics (F1-score and ROC-AUC).
            
        Logic:
            1. Extract predictions and labels from EvalPrediction.
            2. Call multi_label_metrics to compute the metrics.
            3. Return the computed metrics.
        '''

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.multi_label_metrics(predictions=preds,
                                          labels=p.label_ids)
        
        return result
    

    

        

        







        
