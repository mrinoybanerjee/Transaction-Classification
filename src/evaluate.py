import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a given dataset for accuracy, precision, recall, and F1 score.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader for the dataset to evaluate on.
        device (torch.device): The device to perform evaluation on.

    Returns:
        dict: A dictionary containing the metrics 'accuracy', 'precision', 'recall', and 'f1'.
    """
    model.eval()  # Ensure the model is in evaluation mode.
    total_eval_accuracy = 0

    # To accumulate the predictions and true labels of all batches
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)  # Move batch to device
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():  # Do not calculate gradients
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        # Accumulate the predictions and true labels
        all_predictions.extend(predictions)
        all_true_labels.extend(labels_flat)

        total_eval_accuracy += np.sum(predictions == labels_flat) / len(labels_flat)

    # Compute the average accuracy
    avg_accuracy = total_eval_accuracy / len(dataloader)

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')

    metrics = {
        'accuracy': avg_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    # Optionally, print the metrics
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}: {value:.4f}')

    return metrics
