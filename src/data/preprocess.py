import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

def extract_resolution(resolution_str):
    """Extract width and height from resolution string."""
    if isinstance(resolution_str, str):
        match = re.search(r'(\d+)\s*x\s*(\d+)', resolution_str)
        if match:
            width, height = map(int, match.groups())
            return width, height
    return None, None

def convert_to_gb(size_str):
    """Convert storage size to GB."""
    if not isinstance(size_str, str):
        return None
    
    size_str = size_str.lower()
    if 'tb' in size_str:
        return float(re.search(r'([\d.]+)', size_str).group(1)) * 1024
    elif 'gb' in size_str:
        return float(re.search(r'([\d.]+)', size_str).group(1))
    return None

def extract_gpu_memory(gpu_str):
    """Extract GPU memory in GB from GPU string."""
    if not isinstance(gpu_str, str):
        return None
    
    gpu_str = gpu_str.lower()
    match = re.search(r'(\d+)\s*gb', gpu_str)
    if match:
        return float(match.group(1))
    return None

def extract_gpu_brand(gpu_str):
    """Extract GPU brand from GPU string."""
    if not isinstance(gpu_str, str):
        return 'Unknown'
    
    gpu_str = gpu_str.lower()
    if 'nvidia' in gpu_str:
        return 'NVIDIA'
    elif 'amd' in gpu_str or 'radeon' in gpu_str:
        return 'AMD'
    elif 'intel' in gpu_str:
        return 'Intel'
    return 'Other'

def preprocess_data(df):
    """Main preprocessing function."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Drop unnamed columns
    df = df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, errors='ignore')
    
    # Handle missing values
    df = df.dropna()
    
    # Process resolution (already have width and height)
    df['resolution'] = df['resolution_width'] * df['resolution_height']
    
    # Convert ROM to GB
    df['storage_gb'] = df['ROM'].map(convert_to_gb)
    
    # Extract RAM (assuming it's in format like "8GB")
    df['ram_gb'] = df['Ram'].str.extract(r'(\d+)').astype(float)
    
    # Process GPU information
    df['gpu_memory_gb'] = df['GPU'].map(extract_gpu_memory)
    df['gpu_brand'] = df['GPU'].map(extract_gpu_brand)
    
    # Fill missing GPU memory with 0 (integrated graphics)
    df['gpu_memory_gb'] = df['gpu_memory_gb'].fillna(0)
    
    # Print price statistics before encoding
    print("\nPrice Statistics (before processing):")
    print(df['price'].describe())
    
    # Handle categorical variables
    le = LabelEncoder()
    categorical_columns = ['brand', 'processor', 'OS', 'Ram_type', 'ROM_type', 'gpu_brand']
    for col in categorical_columns:
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    
    # Drop original columns that have been transformed
    columns_to_drop = ['ROM', 'Ram', 'resolution_width', 'resolution_height', 'GPU']
    df = df.drop(columns=columns_to_drop)
    
    # Print final price statistics
    print("\nPrice Statistics (after processing):")
    print(df['price'].describe())
    
    return df

def load_and_preprocess(file_path):
    """Load and preprocess the dataset."""
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Apply preprocessing
    df_processed = preprocess_data(df)
    
    # Save processed data
    processed_path = 'data/processed/processed_laptop_data.csv'
    df_processed.to_csv(processed_path, index=False)
    
    return df_processed

if __name__ == "__main__":
    # Path to your raw data
    raw_data_path = 'data/raw/laptop_data.csv'
    
    try:
        processed_df = load_and_preprocess(raw_data_path)
        print("Data preprocessing completed successfully!")
        print(f"Processed data shape: {processed_df.shape}")
        print("\nProcessed columns:")
        print(processed_df.columns.tolist())
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}") 