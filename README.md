# Bank-BERT 💰

## Overview
Bank-BERT leverages a fine-tuned BERT model from Hugging Face's transformers library to efficiently categorize bank transactions. This versatile application supports both script-based batch processing and an interactive web interface via Streamlit, which offers user-friendly visual analytics on transaction categories.

## Features
- **Transaction Categorization**: Utilizes a fine-tuned BERT model to accurately classify bank transactions into predefined categories.
- **Data Visualization**: Provides dynamic visual analytics, including pie charts of transaction categories, bar charts detailing spending by category, and line charts to visualize spending trends.
- **File Handling**: Facilitates the processing of uploaded CSV files through a web interface, with options for downloading categorized data.
- **Model Management**: Efficiently manages the lifecycle of the model including training, saving, loading, and inference.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip for Python package management

### Dependencies
Install the necessary Python packages with pip:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Script
For batch processing via the command line, navigate to the source directory and run:

```bash
python main.py
```
Follow the prompts to specify the path to your dataset.

### Running the Web App
To start the Streamlit web application, execute:

```bash
streamlit run app.py
```
Access the app through the URL provided in your terminal to interact with the features.

## Project Structure

```plaintext
.
├── README.md
├── app.py
├── data
│   └── raw
│       └── transactions.csv
├── models
│   ├── bert_model.pth
│   └── label_encoder.pkl
├── notebooks
│   ├── Data Exploration.ipynb
│   └── Model Training.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── data_create.py
│   └── preprocess.py
└── utils.py
```

## Documentation

### Modules
- **data_create.py**: Functions for loading and preprocessing data, including tokenizing descriptions and encoding categories. Prepares the data for the training pipeline.
- **preprocess.py**: Handles processing of transaction descriptions using the Hugging Face API and manages label encoding.
- **model.py**: Manages BERT model configuration, training, saving, and uploading to Hugging Face Hub.
- **main.py**: CLI interface for processing transactions through the model.
- **app.py**: Streamlit application for interactive file upload, transaction categorization, and viewing visual analytics.

### Functions
- `load_data(file_path)`: Loads and preprocesses the transaction data from a CSV file.
- `encode_categories(df)`: Encodes transaction categories for modeling.
- `tokenize_data(df)`: Tokenizes transaction descriptions for BERT processing.
- `setup_model(num_labels)`: Initializes the BERT model for sequence classification.
- `train_model(model, train_dataset, val_dataset)`: Trains the BERT model and evaluates it using the training and validation datasets.
- `save_and_push(model, tokenizer, model_name, api_token)`: Saves the model and tokenizer and pushes them to the Hugging Face Hub.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with your enhancements. For major changes, open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contact
For support, contact [mrinoybanerjee@gmail.com](mailto:mrinoybanerjee@gmail.com).
```
