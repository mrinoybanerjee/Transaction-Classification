import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

class BertForSequenceClassificationCustom(nn.Module):
    """
    A custom BERT model for sequence classification tasks.

    Attributes:
        bert (BertModel): The pre-trained BERT model.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        classifier (nn.Linear): Linear layer for classification.
    """
    def __init__(self, num_labels):
        """
        Initializes the model with a specified number of labels for classification.

        Args:
            num_labels (int): The number of labels in the classification task.
        """
        super(BertForSequenceClassificationCustom, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input IDs for BERT.
            attention_mask (torch.Tensor, optional): Attention mask for handling padding.

        Returns:
            torch.Tensor: The logits predicted by the model.
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train(model, train_dataloader, validation_dataloader, device, epochs, learning_rate=2e-5, eps=1e-8):
    """
    Trains the BERT model.

    Args:
        model (BertForSequenceClassificationCustom): The model to be trained.
        train_dataloader (DataLoader): The DataLoader for training data.
        validation_dataloader (DataLoader): The DataLoader for validation data.
        device (torch.device): The device to train on.
        epochs (int): The number of epochs to train for.
        learning_rate (float): Learning rate for the optimizer.
        eps (float): Epsilon value for the optimizer.
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

        print(f'Epoch {epoch_i + 1} of {epochs} complete. Avg Loss: {total_loss / len(train_dataloader):.2f}')

def save_model(model, file_path):
    """
    Saves the model to a specified file path.

    Args:
        model (BertForSequenceClassificationCustom): The model to save.
        file_path (str): The path to save the model.
    """
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

def load_model(model_path, num_labels, device):
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, dataloader, device):
    """
    Predicts categories for given data using the trained model.

    Args:
        model (BertForSequenceClassificationCustom): The trained model.
        dataloader (DataLoader): DataLoader for the data to predict.
        device (torch.device): The device to predict on.

    Returns:
        List[int]: List of prediction indices.
    """
    predictions = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.detach().cpu().numpy()
            preds = logits.argmax(axis=1).tolist()
            predictions.extend(preds)
    
    return predictions
