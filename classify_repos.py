import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Basic EDA: Print shape before cleaning
    print(f"Initial shape: {df.shape}")
    
    # Remove rows with empty dependencies or label
    # Assuming empty strings or NaN need to be removed
    df = df.dropna(subset=['dependencies', 'label'])
    df = df[df['dependencies'].str.strip() != '']
    df = df[df['label'].str.strip() != '']
    
    print(f"Shape after cleaning: {df.shape}")
    return df

def parse_dependencies(dep_str):
    # Helper to convert string representation of list to actual list
    # Adjust this based on actual format (e.g., "['numpy', 'pandas']" or "numpy, pandas")
    try:
        # If it looks like a python list string
        if dep_str.strip().startswith('['):
            return ast.literal_eval(dep_str)
        # If it is comma separated
        else:
            return [x.strip() for x in dep_str.split(',')]
    except:
        return []

def main():
    main_file = 'joss_all_language_depends_label.csv'
    lookup_file = 'dependency_counts_with_labels.csv'
    
    # 1. Load and Clean Main Data
    df = load_and_clean_data(main_file)
    
    # 2. Load Lookup Data to define vocabulary
    print(f"Loading lookup table from {lookup_file}...")
    try:
        lookup_df = pd.read_csv(lookup_file)
        # Assuming the lookup file has a column like 'dependency' that lists unique valid dependencies
        # If the column name is different, adjust 'dependency' below
        valid_dependencies = set(lookup_df['dependency'].unique())
    except Exception as e:
        print(f"Error loading lookup file: {e}")
        return

    # 3. Feature Extraction
    print("Extracting features...")
    
    # Convert dependency strings to lists
    df['dep_list'] = df['dependencies'].apply(parse_dependencies)
    
    # Filter dependencies in the main dataframe to only include those in the lookup table
    df['filtered_deps'] = df['dep_list'].apply(lambda x: [dep for dep in x if dep in valid_dependencies])
    
    # Use MultiLabelBinarizer to create a binary matrix (One-Hot Encoding style for lists)
    # We fit only on the valid_dependencies to ensure consistent feature space
    mlb = MultiLabelBinarizer(classes=sorted(list(valid_dependencies)))
    
    # Transform the filtered dependencies into feature vectors
    X = mlb.fit_transform(df['filtered_deps'])
    y = df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    
    # 4. Train/Test Split (70/30)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # 5. Model Training
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Evaluation
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
