import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_and_save_model():
    """
    Loads the Housing.csv dataset, trains a simple linear regression model
    to predict the 'price', and saves the trained model as a pickle file.
    """
    # Define the file path
    dataset_path = os.path.join('Datasets', 'Housing.csv')
    model_output_path = 'housing_model.pkl'

    # 1. Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found. Make sure the file is in the 'Datasets' folder.")
        return

    # 2. Preprocess the data
    print("Preprocessing data...")
    # Convert categorical variables to dummy variables (one-hot encoding)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # Define features (X) and target (y)
    X = df_dummies.drop('price', axis=1)
    y = df_dummies['price']

    # 3. Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train the model
    print("Training a Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Save the trained model
    print(f"Saving the trained model to {model_output_path}...")
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model successfully saved to {model_output_path}.")
    print("\nYou can now use this file to test the model performance evaluation endpoint.")

if __name__ == '__main__':
    train_and_save_model()
