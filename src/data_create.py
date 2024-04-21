import torch
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_for_training(file_path):
    """
    Loads data from a CSV file and preprocesses it for training or inference. Filters out negative transactions.

    Args:
        file_path (str): Path to the CSV file.
        for_training (bool): If True, processes data for training including encoding categories.

    Returns:
        tuple: A tuple containing the DataFrame and possibly category labels if for training.
    """
    try:
        df = pd.read_csv(file_path)
        df = df[df['Description'].str.upper() != 'DESCRIPTION']  # Remove any rows that are header rows repeated.
        df = df[df['Amount'] >= 0]  # Filter out transactions with negative amounts.

        if 'Description' not in df.columns:
            raise ValueError("Dataset must contain a 'Description' column.")
        
        if 'Category' not in df.columns:
            raise ValueError("For training, the dataset must contain a 'Category' column.")
        label_encoder = LabelEncoder()
        df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
        # Save the encoder for later use
        with open('./models/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        return df, label_encoder.classes_
    
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        raise


def tokenize_data(df):
    """
    Tokenizes text data in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing text to be tokenized.

    Returns:
        tuple: Tensors of input IDs and attention masks.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []

    for desc in df['Description']:
        encoded_dict = tokenizer.encode_plus(
            desc, add_special_tokens=True, max_length=64, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def create_data_loaders(input_ids, attention_masks, labels, batch_size=32):
    """
    Creates DataLoader for training and validation.

    Args:
        input_ids (torch.Tensor): Input IDs from tokenizer.
        attention_masks (torch.Tensor): Attention masks from tokenizer.
        labels (torch.Tensor): Encoded labels for the inputs.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: DataLoaders for both training and validation datasets.
    """
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, random_state=42, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, random_state=42, test_size=0.1)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    return train_dataloader, validation_dataloader
