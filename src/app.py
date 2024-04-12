import streamlit as st
import pandas as pd
import torch
from data_create import load_and_preprocess_data, tokenize_data
from model import load_model, predict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import os
from io import BytesIO

def process_file(uploaded_file, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ['Miscellaneous', 'Groceries', 'Technology', 'Food', 'Utilities', 'Travel', 'Entertainment', 'Transportation', 'Services', 
                  'Clothing and Accessories', 'Health and Wellness', 'Personal Care', 'Membership fees', 'Rewards', 'Shipping', 'Income', 'Housing', 
                  'Communications', 'Education', 'Insurance', 'Credit Card Fee', 'Investment', 'Advertising/Marketing']

    # Assuming uploaded_file is a CSV
    df = pd.read_csv(uploaded_file)
    input_ids, attention_masks = tokenize_data(df)
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

    model = load_model(model_path, len(categories), device)
    prediction_indices = predict(model, prediction_dataloader, device)
    predicted_categories = [categories[pred] for pred in prediction_indices]

    df['Predicted_Category'] = predicted_categories
    return df

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Create a bytes buffer for the to-be-downloaded file
    buffer = BytesIO()
    buffer.write(object_to_download.encode())
    buffer.seek(0)

    return st.download_button(label=download_link_text,
                              data=buffer,
                              file_name=download_filename,
                              mime='text/csv')

st.title('Bank Transaction Categorization')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
model_path = st.text_input("Enter the path to your pre-trained model:", "")

if st.button('Categorize Transactions'):
    if uploaded_file is not None and model_path:
        processed_df = process_file(uploaded_file, model_path)
        st.write(processed_df.head())  # Display a preview of the processed DataFrame
        download_link(processed_df, 'categorized_transactions.csv', 'Download Categorized Transactions')
    else:
        st.error("Please upload a file and enter the model path.")
