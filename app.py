import streamlit as st
import pandas as pd
import torch
from src.data_create import tokenize_data
from src.model import load_model, predict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from io import BytesIO
import pickle
import logging
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    df = pd.read_csv(uploaded_file)

    # Convert Amount to numeric and exclude negative transactions
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df[df['Amount'] >= 0]

    try:
        with open('./models/label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)
    except FileNotFoundError:
        st.error("Label encoder file not found. Please ensure it is in the correct path.")
        return pd.DataFrame()  # Return an empty DataFrame to prevent further execution

    input_ids, attention_masks = tokenize_data(df)
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

    model = load_model(len(label_encoder.classes_), device)
    prediction_indices = predict(model, prediction_dataloader, device)
    df['Predicted_Category'] = label_encoder.inverse_transform(prediction_indices)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime object
    return df

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the specified DataFrame as a CSV file.

    Args:
        object_to_download (pd.DataFrame): The DataFrame to be downloaded.
        download_filename (str): The name of the file to be downloaded.
        download_link_text (str): The text displayed on the download button.

    Returns:
        Streamlit download button: A button that when clicked downloads the CSV file.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    buffer = BytesIO()
    buffer.write(object_to_download.encode())
    buffer.seek(0)
    return st.download_button(label=download_link_text, data=buffer, file_name=download_filename, mime='text/csv')

def create_category_pie_chart(df):
    """
    Generate a pie chart displaying the breakdown of transaction categories.

    Args:
        df (pd.DataFrame): DataFrame containing the 'Predicted_Category' column.
    """
    category_counts = df['Predicted_Category'].value_counts()
    fig = px.pie(values=category_counts.values, names=category_counts.index, title='Number of Transactions by Category')
    st.plotly_chart(fig)

def create_spending_by_category_bar_chart(df):
    """
    Generate a bar chart showing total spending per transaction category.

    Args:
        df (pd.DataFrame): DataFrame containing 'Predicted_Category' and 'Amount' columns.
    """
    category_spending = df.groupby('Predicted_Category')['Amount'].sum().sort_values(ascending=False)
    fig = px.bar(x=category_spending.index, y=category_spending.values, labels={'x': 'Category', 'y': 'Amount'},
                 title='Dollars Spent by Category')
    st.plotly_chart(fig)

def create_temporal_spending_trend(df, time_freq='M'):
    """
    Create a line chart showing spending trends over time, segmented by month or year.

    Args:
        df (pd.DataFrame): The DataFrame containing transaction data with a 'Date' field.
        time_freq (str): 'M' for monthly and 'Y' for yearly trends.
    """
    df['Period'] = df['Date'].dt.to_period(time_freq)
    temporal_spending = df.groupby('Period')['Amount'].sum()
    title = 'Monthly Spending Trend' if time_freq == 'M' else 'Yearly Spending Trend'
    fig = px.line(x=temporal_spending.index.astype(str), y=temporal_spending.values,
                  labels={'x': 'Period', 'y': 'Total Spending'}, title=title)
    st.plotly_chart(fig)

st.title('Bank-BERT 💰')
st.write('This app categorizes your bank transactions using a fine-tuned BERT model and provides personalized spend analytics.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if st.button('Categorize Transactions'):
    if uploaded_file is not None:
        processed_df = process_file(uploaded_file)
        if not processed_df.empty:
            st.write(processed_df.head())  # Display a preview of the processed DataFrame
            download_link(processed_df, 'categorized_transactions.csv', 'Download Categorized Transactions')
            
            # Visualization
            st.header('Visual Analytics')
            create_category_pie_chart(processed_df)
            create_spending_by_category_bar_chart(processed_df)
            create_temporal_spending_trend(processed_df, 'M')
            create_temporal_spending_trend(processed_df, 'Y')
        else:
            st.error("Failed to process the file or no data to display.")
    else:
        st.error("Please upload a file and enter the model path.")
