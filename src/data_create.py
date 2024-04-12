import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizer

def load_and_preprocess_data(file_path, for_training=True):
    """
    Loads and preprocesses the dataset from a CSV file. If the 'Category' column is missing,
    it prepares the data for inference.

    Args:
        file_path (str): The path to the CSV file containing the dataset.
        for_training (bool): Flag indicating whether the data is being loaded for training
                             or inference. If True, expects 'Category' column to be present.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
        np.ndarray (optional): Array of unique class labels if for_training is True.
    """
    df = pd.read_csv(file_path)
    
    # Check if 'Description' column exists
    if 'Description' not in df.columns:
        raise ValueError("The dataset must contain a 'Description' column.")
    
    # Remove any rows that might have replicated header information
    df = df[df['Description'] != 'Description']

    if for_training:
        # For training, ensure 'Category' column exists
        if 'Category' not in df.columns:
            raise ValueError("For training, the dataset must contain a 'Category' column.")
        label_encoder = LabelEncoder()
        df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
        return df, label_encoder.classes_
    else:
        # For inference, return the DataFrame without category encoding
        return df

def tokenize_data(df):
    """
    Tokenizes the descriptions in the dataset.

    Args:
        df (pd.DataFrame): The DataFrame with a 'Description' column to tokenize.

    Returns:
        torch.Tensor: Concatenated token IDs for all descriptions.
        torch.Tensor: Concatenated attention masks for all descriptions.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []
    
    for desc in df['Description']:
        encoded_dict = tokenizer.encode_plus(
            desc,                              # Sentence to encode.
            add_special_tokens=True,          # Add '[CLS]' and '[SEP]'.
            max_length=64,                    # Pad & truncate all sentences.
            padding='max_length',             # Pad to max length
            truncation=True,                  # Explicitly truncate to max length
            return_attention_mask=True,       # Construct attention masks.
            return_tensors='pt',              # Return PyTorch tensors.
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

def create_data_loaders(input_ids, attention_masks, labels, batch_size=32):
    """
    Creates train and validation DataLoader objects.

    Args:
        input_ids (torch.Tensor): Tensor of token IDs.
        attention_masks (torch.Tensor): Tensor of attention masks.
        labels (torch.Tensor): Tensor of labels.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the training set.
        DataLoader: DataLoader for the validation set.
    """
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.1)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    return train_dataloader, validation_dataloader
