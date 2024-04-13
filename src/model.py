import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import logging

# Setup logging
logger = logging.getLogger(__name__)

class BertForSequenceClassificationCustom(nn.Module):
    """
    A custom implementation of BERT for sequence classification tasks.
    
    Attributes:
        bert (BertModel): The pre-trained BERT model from Hugging Face's Transformers.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        classifier (nn.Linear): A linear layer for classification that maps from the hidden states to the output labels.
    
    Args:
        num_labels (int): The number of labels in the classification task (size of the output layer).
    """
    def __init__(self, num_labels):
        super(BertForSequenceClassificationCustom, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs to be fed to the BERT model.
            attention_mask (torch.Tensor, optional): Tensor representing attention masks to avoid focusing on padding.
        
        Returns:
            torch.Tensor: Output logits from the classifier.
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # We use the pooled output from BERT that represents the [CLS] token.
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train(model, train_dataloader, validation_dataloader, device, epochs, learning_rate=2e-5, eps=1e-8):
    """
    Trains the BERT model using the given data loaders and hyperparameters.

    Args:
        model (BertForSequenceClassificationCustom): The model to be trained.
        train_dataloader (DataLoader): DataLoader for the training data.
        validation_dataloader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to train the model on (CPU or GPU).
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        eps (float): Epsilon for the Adam optimizer (helps with numerical stability).
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=eps)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch_i in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()
            logits = model(b_input_ids, attention_mask=b_input_mask)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.num_labels), b_labels.view(-1))

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        logger.info(f'Epoch {epoch_i + 1} of {epochs} complete. Avg Loss: {total_loss / len(train_dataloader):.2f}')

    # Save the model after training
    save_model(model, './models/bert_custom_model.pth')


def save_model(model, file_path):
    """
    Saves the model to a specified file path.

    Args:
        model (BertForSequenceClassificationCustom): The model to save.
        file_path (str): Path where the model will be saved.
    """
    torch.save(model.state_dict(), file_path)
    logger.info(f'Model saved to {file_path}')

def load_model(num_labels, device):
    """
    Loads a pre-trained model from a specified file path.

    Args:
        model_path (str): The path to the model file.
        num_labels (int): The number of labels in the model's classification layer.
        device (torch.device): The device to load the model onto.

    Returns:
        BertForSequenceClassificationCustom: The loaded model.
    """
    model = BertForSequenceClassificationCustom(num_labels)
    model_path = './models/Fine Tuned BERT Model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, dataloader, device):
    """
    Performs prediction using the provided model and dataloader.

    Args:
        model (BertForSequenceClassificationCustom): The trained model.
        dataloader (DataLoader): DataLoader containing the input data for prediction.
        device (torch.device): The device on which to perform the prediction.

    Returns:
        list: Predictions for the input data.
    """
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_input_mask = batch[:2]  # Assuming no labels are provided
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.detach().cpu().numpy()
            preds = logits.argmax(axis=1).tolist()
            predictions.extend(preds)
    return predictions
