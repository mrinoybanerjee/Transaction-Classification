import pandas as pd
from transformers import BertTokenizer
import pickle
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Load and preprocess the dataset.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    df = df[df['Description'] != 'Description']  # Filter out invalid rows
    df['Description'] = df['Description'].str.lower()  # Convert to lowercase
    # removing negative transactions
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df[df['Amount'] >= 0]
    # removing categories with less than 10 transactions
    category_counts = df['Category'].value_counts()
    df = df[df['Category'].isin(category_counts[category_counts >= 10].index)]
    return df

def encode_categories(df):
    """
    Encode category labels in the dataset.
    Args:
        df (pd.DataFrame): DataFrame with 'Category' column to encode.
    Returns:
        pd.DataFrame: DataFrame with encoded categories.
    """
    label_encoder = LabelEncoder()
    df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
    # Save the encoder to disk for later use
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    return df


def tokenize_data(df, model_name='bert-base-uncased'):
    """
    Tokenize descriptions in the dataset.
    Args:
        df (pd.DataFrame): DataFrame with 'Description' to tokenize.
        model_name (str): Model name for tokenizer initialization.
    Returns:
        tuple: Tuple containing lists of input_ids and attention_masks.
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenized_outputs = df['Description'].apply(
            lambda desc: tokenizer.encode_plus(
                desc,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=64,
                return_attention_mask=True,
                return_tensors='pt'
            )
        )
        input_ids = [output['input_ids'].squeeze(0).tolist() for output in tokenized_outputs]
        attention_masks = [output['attention_mask'].squeeze(0).tolist() for output in tokenized_outputs]
        return input_ids, attention_masks
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return [], []
