import pandas as pd
import os

def load_data():
    file_path = os.path.join("data", "house_prices.csv")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    print("\n Dataset Information:")
    print(df.info())

    print("\n Missing Values:")
    print(df.isnull().sum())

    print("\n First 5 Rows:")
    print(df.head())

    df = df.dropna()

    return df

if __name__ == "__main__":
    df = load_data()
    processed_df = preprocess_data(df)

    print("\n Preprocessing completed successfully!")