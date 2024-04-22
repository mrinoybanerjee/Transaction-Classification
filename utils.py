from io import BytesIO
import pandas as pd
import streamlit as st
import plotly.express as px


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