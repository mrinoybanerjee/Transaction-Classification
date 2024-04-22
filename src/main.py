from preprocess import preprocess_file

def main():
    """
    Main function to run the transaction categorization pipeline using Hugging Face API.
    """
    print("\nWelcome to Bank-BERT 💰!\n")
    file_path = input("Please enter the path to your dataset CSV file: ")

    df = preprocess_file(file_path)

    result_file_path = "categorized_transactions.csv"
    df.to_csv(result_file_path, index=False)
    print(f"Processed transactions have been saved to {result_file_path}")

if __name__ == "__main__":
    main()
