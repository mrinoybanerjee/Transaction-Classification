import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def setup_model(num_labels):
    """
    Initialize a BERT model for sequence classification.
    Args:
        num_labels (int): Number of distinct categories to classify.
    Returns:
        BertForSequenceClassification: The initialized model.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

def train_model(model, train_dataset, val_dataset):
    """
    Train the BERT model.
    Args:
        model (BertForSequenceClassification): Model to train.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
    Returns:
        Trainer: The trained model.
    """
    try:
        training_args = TrainingArguments(
            output_dir='../data/results',
            num_train_epochs=15,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='../data/results/logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()
        return trainer
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None


def save_and_push(model, tokenizer, model_name, api_token):
    """
    Save and push the model and tokenizer to the Hugging Face Hub.
    Args:
        model (BertForSequenceClassification): Trained model.
        tokenizer (BertTokenizer): Associated tokenizer.
        model_name (str): Repository name on the Hugging Face Hub.
        api_token (str): Hugging Face authentication token.
    """
    model.push_to_hub(model_name, use_auth_token=api_token)
    tokenizer.push_to_hub(model_name, use_auth_token=api_token)
