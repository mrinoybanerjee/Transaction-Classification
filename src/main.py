import pandas as pd
import torch
from data_create import tokenize_data
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pickle
from model import load_model, predict
import pandas as pd
import os

def process_file(uploaded_file):
    """
    Process the uploaded CSV file by predicting the transaction categories using the trained BERT model.
    It also ensures that transactions with negative amounts are excluded.

    Args:
        uploaded_file: The uploaded CSV file handle.
        model_path (str): The path to the pre-trained BERT model.

    Returns:
        pd.DataFrame: The DataFrame with predicted categories added.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load and preprocess the uploaded file
    df = pd.read_csv(uploaded_file)

    # Convert Amount to numeric and exclude negative transactions
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df[df['Amount'] >= 0]

    try:
        with open('../models/label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)
    except FileNotFoundError:
        print("Label encoder file not found. Please ensure it is in the correct path.")
        return pd.DataFrame()  # Return an empty DataFrame to prevent further execution

    # Tokenize the data
    input_ids, attention_masks = tokenize_data(df)
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

    # Load the pre-trained model
    model = load_model(len(label_encoder.classes_), device)
    # Predict transaction categories
    prediction_indices = predict(model, prediction_dataloader, device)
    df['Predicted_Category'] = label_encoder.inverse_transform(prediction_indices)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime object
    return df


def main():
    """
    Main function to run the transaction categorization pipeline locally.
    This function loads a CSV file, processes the data, loads a pre-trained model, predicts transaction categories,
    and saves the results to a new CSV file.
    """
    # Bank-BERT
    print("")
    print("")
    print("Welcome to Bank-BERT 💰!")
    print("")
    file_path = input("Please enter the path to your dataset CSV file: ")

    # Process file and predict categories
    df = process_file(file_path)

    # Save the processed DataFrame to a new CSV
    result_file_path = "../data/categorized/categorized_transactions.csv"
    df.to_csv(result_file_path, index=False)
    print(f"Processed transactions have been saved to {result_file_path}")

if __name__ == "__main__":
    main()
