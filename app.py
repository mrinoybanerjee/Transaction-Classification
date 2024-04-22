import streamlit as st
from src.preprocess import preprocess_file
from utils import download_link, create_category_pie_chart, create_spending_by_category_bar_chart, create_temporal_spending_trend


st.title('Bank-BERT 💰')
st.write('This app categorizes your bank transactions using a fine-tuned BERT model and provides personalized spend analytics.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if st.button('Categorize Transactions'):
    if uploaded_file is not None:
        processed_df = preprocess_file(uploaded_file)
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
