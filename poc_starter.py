"""
AI Pre Launch Prediction Tool - POC Starter Script
====================================================
MBA7008 Capstone Project | Xuan Zhao

This script handles the first pass of the model:
  1. Load the Kaggle marketing campaign dataset
  2. Inspect and clean the data
  3. Train a gradient boosting model to predict conversion rate
  4. Evaluate the model on a test set
  5. Save the trained model so the Streamlit app can use it later

How to run:
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset
  2. Put the CSV file in the same folder as this script
  3. Update the CSV_FILENAME variable below if the filename is different
  4. Open a terminal in this folder and run:
        pip install pandas scikit-learn joblib
        python poc_starter.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# ============================================================
# CONFIG: change this to match your downloaded file name
# ============================================================
CSV_FILENAME = "marketing_campaign_dataset.csv"
TARGET_COLUMN_GUESSES = ["Conversion_Rate", "Conversion Rate", "conversion_rate", "ConversionRate"]
RANDOM_SEED = 42
SAMPLE_SIZE = 10000  # use a subset for faster training (the full dataset is ~200k rows)


# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
def load_data(filename):
    """Load the CSV and return a pandas DataFrame."""
    if not os.path.exists(filename):
        print(f"ERROR: Could not find '{filename}' in this folder.")
        print("Please download it from Kaggle and place it here, then run again.")
        sys.exit(1)

    print(f"Loading {filename} ...")
    df = pd.read_csv(filename)
    print(f"   Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


# ============================================================
# STEP 2: INSPECT THE DATA
# ============================================================
def inspect_data(df):
    """Print a quick summary so we can see what we are working with."""
    print("\n" + "=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    print("\nColumn names:")
    for col in df.columns:
        print(f"   - {col} ({df[col].dtype})")

    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"   - {col}: {count} missing")
    if missing.sum() == 0:
        print("   None! Dataset is clean.")


# ============================================================
# STEP 3: FIND THE TARGET COLUMN
# ============================================================
def find_target_column(df):
    """The target column might have slightly different names. Try a few."""
    for guess in TARGET_COLUMN_GUESSES:
        if guess in df.columns:
            print(f"\nTarget column found: '{guess}'")
            return guess

    # If none of our guesses match, ask the user
    print("\nCould not auto-detect the target column.")
    print("Please look at the column list above and pick the conversion rate column.")
    print("Then update TARGET_COLUMN_GUESSES at the top of this script.")
    sys.exit(1)


# ============================================================
# STEP 4: CLEAN AND PREPARE
# ============================================================
def clean_and_prepare(df, target_col):
    """Drop rows with missing target, encode categories, return X and y."""
    print("\n" + "=" * 60)
    print("CLEANING AND PREPARING")
    print("=" * 60)

    # Sample down to a manageable size for faster training
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Sampled down to {SAMPLE_SIZE:,} rows for faster training")

    # If target is stored as a percentage string (like "7%"), convert to a number
    if df[target_col].dtype == "object" or pd.api.types.is_string_dtype(df[target_col]):
        # Only attempt conversion if values look like text
        if df[target_col].astype(str).str.contains("%", na=False).any() or \
           not pd.api.types.is_numeric_dtype(df[target_col]):
            print(f"Converting '{target_col}' from text to number...")
            df[target_col] = (
                df[target_col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .astype(float)
            )

    # Drop any rows where the target itself is missing
    before = len(df)
    df = df.dropna(subset=[target_col])
    print(f"Dropped {before - len(df)} rows with missing target")

    # Drop columns that are clearly identifiers and not predictive
    drop_candidates = ["Campaign_ID", "Campaign ID", "ID", "Date", "Customer_ID"]
    for col in drop_candidates:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"   Dropped identifier column: {col}")

    # Encode categorical fields (text fields like "Email", "Display", "Fashionistas")
    print("\nEncoding categorical fields...")
    label_encoders = {}
    for col in df.columns:
        # Use a robust check that works across pandas versions
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            # Normalize text first (lowercase, strip whitespace) to handle
            # the messy "Facebook" vs "facebook" vs "FB" type problem
            df[col] = df[col].astype(str).str.lower().str.strip()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"   - {col}: {len(le.classes_)} unique values encoded")

    # Fill any remaining missing numeric values with the column median
    df = df.fillna(df.median(numeric_only=True))

    # Split into features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nReady for training:")
    print(f"   Features (X): {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"   Target (y):   {y.shape[0]:,} rows")
    print(f"   Target stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

    return X, y, label_encoders


# ============================================================
# STEP 5: TRAIN AND EVALUATE THE MODEL
# ============================================================
def train_and_evaluate(X, y):
    """Split 80/20, train gradient boosting, report R-squared and MAE."""
    print("\n" + "=" * 60)
    print("TRAINING THE MODEL")
    print("=" * 60)

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Training set: {len(X_train):,} rows")
    print(f"Test set:     {len(X_test):,} rows")

    # Train a gradient boosting regressor
    print("\nTraining gradient boosting model (this takes about 30 seconds)...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    print("Training complete")

    # Evaluate on the test set
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nR-squared: {r2:.3f}  (target: above 0.70)")
    print(f"MAE:       {mae:.4f}")

    if r2 >= 0.70:
        print("\nMet the target. Model is ready for the Streamlit interface.")
    elif r2 >= 0.50:
        print("\nBelow target but workable. Consider feature engineering or backup model.")
    else:
        print("\nLow R-squared. The dataset may not have enough signal to beat random.")
        print("This is normal for synthetic Kaggle data. Honest reporting is fine for the POC.")

    # Show the top 5 features that drove the prediction (useful for the trust/explainability story)
    print("\nTop 5 most important features:")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feature, importance in importances.head(5).items():
        print(f"   {feature}: {importance:.3f}")

    return model, r2, mae


# ============================================================
# STEP 6: SAVE THE MODEL
# ============================================================
def save_model(model, label_encoders, feature_names):
    """Save everything the Streamlit app will need to make predictions."""
    output = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_names": list(feature_names),
    }
    joblib.dump(output, "model.pkl")
    print("\nSaved model to 'model.pkl'")
    print("   The Streamlit app will load this file to make predictions.")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AI Pre Launch Prediction Tool - POC Starter")
    print("=" * 60)

    df = load_data(CSV_FILENAME)
    inspect_data(df)
    target_col = find_target_column(df)
    X, y, label_encoders = clean_and_prepare(df, target_col)
    model, r2, mae = train_and_evaluate(X, y)
    save_model(model, label_encoders, X.columns)

    print("\n" + "=" * 60)
    print("DONE. Next step: build the Streamlit interface that loads model.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
