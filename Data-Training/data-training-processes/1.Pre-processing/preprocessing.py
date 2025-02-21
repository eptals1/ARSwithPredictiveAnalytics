import re
import pandas as pd
from typing import Union, List

def preprocess_text(text: str) -> str:
    """
    Preprocess text by applying basic cleaning operations
    
    Args:
        text: Input text string
    
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    return text

def load_and_preprocess_csv(csv_path: str, text_column: str = 'text') -> pd.DataFrame:
    """
    Load CSV file and preprocess the text column
    
    Args:
        csv_path: Path to the CSV file
        text_column: Name of the column containing text to preprocess
    
    Returns:
        DataFrame with original and preprocessed text
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Check if text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV file")
    
    # Apply preprocessing to text column
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)
    
    return df

if __name__ == "__main__":
    # Example usage
    csv_path = "dataset/2.extracted-text/extracted-text-resumes.csv"
    
    try:
        # Load and preprocess the resume data
        df = load_and_preprocess_csv(csv_path)
        
        # Save preprocessed data
        output_path = "dataset/3.pre-processed/csv-files/resumes.csv"
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        
        # Show sample of preprocessing results
        print("\nSample of preprocessing results:")
        for _, row in df.head().iterrows():
            print(f"\nOriginal: {row['text'][:100]}...")
            print(f"Preprocessed: {row['preprocessed_text'][:100]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")
