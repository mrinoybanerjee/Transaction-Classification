# Bank-BERT 💰

## Overview
Bank-BERT categorizes bank transactions using a fine-tuned BERT-base model from Hugging Face's transformers library. This application can be run as a script for batch processing or as a web app using Streamlit, which provides a user-friendly interface and visual analytics of categorized transactions.

## Features
- **Transaction Categorization**: Classifies bank transactions into predefined categories using a BERT model.
- **Data Visualization**: Displays visual analytics including pie charts of transaction categories, bar charts of spending by category, and line charts of spending trends over time.
- **File Handling**: Processes uploaded CSV files through a web interface and gives the option to downloaded categorized csv file.
- **Model Management**: Handles model training, saving, loading, and inference within the Python ecosystem.

## Installation

### Prerequisites
- Python 3.8 or later
- pip for managing Python packages

### Dependencies
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Script
To run the categorization pipeline via the command line, navigate to the src directory and execute:

```bash
python main.py
```
Follow the on-screen prompts to provide the path to your dataset.

### Running the Web App
To launch the Streamlit web application, navigate to the src directory and run:
```bash
streamlit run app.py
```
Navigate to the provided URL in your web browser to interact with the application.

## Project Structure
```plaintext
Bank-BERT/
├── README.md
├── data
│   ├── categorized
│   │   ├── Amex Categorized Raw.csv
│   │   ├── Synthetic transactions data.csv
│   │   └── categorized_transactions.csv
│   └── raw
│       └── all_amex.csv
├── models
│   ├── Fine Tuned BERT Model.pth
│   └── label_encoder.pkl
├── notebooks
│   ├── Deep Learning Approach.ipynb
│   ├── Evaluation.ipynb
│   └── Non Deep Learning Approach.ipynb
├── requirements.txt
└── src
    ├── __pycache__
    │   ├── data_create.cpython-311.pyc
    │   └── model.cpython-311.pyc
    ├── app.py
    ├── data_create.py
    ├── evaluate.py
    ├── main.py
    └── model.py

```

## Documentation

### Modules
- **data_create.py**: Contains functions for loading, preprocessing, and tokenizing data.
- **model.py**: Defines the BERT-based neural network architecture, training, and prediction logic.
- **main.py**: Provides a CLI-based approach to process a CSV file and categorize transactions.
- **app.py**: Streamlit application that allows users to upload a CSV file, categorize transactions, and view visual analytics.

### Functions
- `load_and_preprocess_for_training(file_path)`: Loads and preprocesses the CSV data for the training pipeline.
- `tokenize_data(df)`: Tokenizes text data in the DataFrame.
- `create_data_loaders(input_ids, attention_masks, labels, batch_size)`: Creates DataLoader instances for model training.
- `BertForSequenceClassificationCustom`: Custom BERT model for sequence classification tasks.
- `train(model, train_dataloader, validation_dataloader, device, epochs, learning_rate, eps)`: Trains the BERT model.
- `predict(model, dataloader, device)`: Predicts categories using the trained model.
- `process_file(uploaded_file)`: Processes the uploaded CSV file and categorizes transactions to run inference.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with your enhancements. For major changes, please open an issue first to discuss what you would like to change.

Ensure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contact
For support, contact [mrinoybanerjee@gmail.com](mailto:mrinoybanerjee@gmail.com).