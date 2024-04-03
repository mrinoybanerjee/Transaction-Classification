import torch
from data_create import load_and_preprocess_data, tokenize_data
from model import load_model, predict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def main():
    """
    Main function to categorize transactions from a CSV file using a pre-trained model and save the results.
    """
    # User input for file paths
    file_path = input("Please enter the path to your dataset CSV file: ")
    model_path = input("Please enter the path to your pre-trained model: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardcoded list of categories the model is fine-tuned on
    categories = ['Miscellaneous', 'Groceries', 'Technology', 'Food', 'Utilities', 'Travel', 'Entertainment', 'Transportation', 'Services', 
                  'Clothing and Accessories', 'Health and Wellness', 'Personal Care', 'Membership fees', 'Rewards', 'Shipping', 'Income', 'Housing', 
                  'Communications', 'Education', 'Insurance', 'Credit Card Fee', 'Investment', 'Advertising/Marketing']

    # Load and preprocess data for inference
    df = load_and_preprocess_data(file_path, for_training=False)
    input_ids, attention_masks = tokenize_data(df)

    # Prepare data for prediction
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

    # Load the pre-trained model, ensuring it's loaded to the correct device
    model = load_model(model_path, len(categories), device)

    # Predict categories
    prediction_indices = predict(model, prediction_dataloader, device)
    predicted_categories = [categories[pred] for pred in prediction_indices]

    # Append predicted categories to DataFrame and save to new CSV
    df['Predicted_Category'] = predicted_categories
    result_file_path = "data/categorized/categorized_transactions.csv"
    df.to_csv(result_file_path, index=False)
    print(f"Processed transactions have been saved to {result_file_path}")

if __name__ == "__main__":
    main()
