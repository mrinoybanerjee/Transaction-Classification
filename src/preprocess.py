import requests
import pickle
import requests
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Description: Utility functions for processing transaction data.
#  - process_description: Process the transaction description using the Hugging Face API to predict the category.
#  - process_labels: Apply label encoder to decode the predicted labels.
#  - process_file: Process the uploaded CSV file by predicting the transaction categories using the Hugging Face inference API.

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://h27fegzq22vqhmsi.us-east-1.aws.endpoints.huggingface.cloud"
HF_HEADERS = {
	"Accept" : "application/json",
	"Authorization": f"Bearer {HF_TOKEN}",
	"Content-Type": "application/json" 
}

def process_description(desc: str) -> str:
    '''
    Process the transaction description using the Hugging Face API to predict the category.

    Args:
        desc: The transaction description to process.
    
    Returns:
        str: The predicted category for the transaction.
    '''
    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": desc}
        )
        response.raise_for_status()  # This will raise an exception for HTTP errors
        data = response.json()
        # Check if the expected key 'label' is in the response and the response is properly structured
        if data and isinstance(data, list) and "label" in data[0]:
            return data[0]["label"]
        else:
            st.error("Unexpected response structure from API")
            return None  # You may choose to handle this case differently depending on your application's needs
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
    except ValueError as e:
        st.error(f"JSON decoding failed: {e}")
        return None

def process_labels(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Apply label encoder to decode the predicted labels
    
    Args:
        df: The DataFrame with predicted categories added.
        
    Returns:
        pd.DataFrame: The DataFrame with predicted categories decoded.
    '''
    # Load the label encoder
    try:
        with open('./models/label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)
    except FileNotFoundError:
        st.error("Label encoder file not found. Please ensure it is in the correct path.")
        return pd.DataFrame()  # Return an empty DataFrame to prevent further execution
    
    # Extract numeric parts from the labels 'LABEL_X' and convert them to integers
    numeric_labels = [int(label.split('_')[1]) for label in df['encoded_label']]
    
    # Use inverse_transform to decode the numeric labels
    df['Predicted_Category'] = label_encoder.inverse_transform(numeric_labels)
    
    # Remove the encoded_label column
    df = df.drop(columns=['encoded_label'])
    return df

def preprocess_file(uploaded_file):
    """
    Preprocess the uploaded CSV file by predicting the transaction categories using the Hugging Face inference API.

    Args:
        uploaded_file: The path to the uploaded CSV file.
        api_url (str): The Hugging Face API endpoint for predictions.
        headers (dict): Authentication headers for the Hugging Face API.

    Returns:
        pd.DataFrame: The DataFrame with predicted categories added.
    """
    df = pd.read_csv(uploaded_file)

    # Convert Amount to numeric and exclude negative transactions
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df[df['Amount'] >= 0]
    # Convert description to lower case to prepare for bert base uncased
    df['Description'] = df['Description'].str.lower()
    df["encoded_label"] = df["Description"].apply(lambda desc: process_description(desc) if pd.notna(desc) else None)

    # Handle potential None values before processing labels
    df = df.dropna(subset=["encoded_label"])
    df = process_labels(df)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime object
    print(df.head())
    return df