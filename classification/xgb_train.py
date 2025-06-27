import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATASET_CSV = "ann_dataset_binary.csv"
MODEL_FILENAME = "binary_interaction_classifier_xgb.json"

def train_binary_xgb():
    """
    Trains a binary XGBoost classifier for 'interacting' vs 'non-interacting'.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(DATASET_CSV)
        logging.info(f"Loaded dataset: {DATASET_CSV} with {len(df)} samples.")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {DATASET_CSV}. Please run data.py first.")
        return

    # 2. Preprocess Data
    feature_columns = ["anchor_dist", "orientation_angle_deg"]
    df.dropna(subset=feature_columns, inplace=True)
    X = df[feature_columns]
    y = df["label"]

    # Encode labels to 0 and 1
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    logging.info(f"Labels encoded. Classes: {class_names} -> {list(range(len(class_names)))}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logging.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # 4. Build and Train XGBoost Model
    # <<< UPDATED: Objective for binary classification >>>
    model = xgb.XGBClassifier(
        objective='binary:logistic', # Use 'binary:logistic' for binary problems
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss', # Common metric for binary classification
        random_state=42
    )
    
    # Create a validation set for early stopping
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    logging.info("Starting XGBoost model training...")
    model.fit(X_train_part, y_train_part, 
              eval_set=[(X_val, y_val)],
              verbose=False)
    logging.info("Training finished.")

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_mcc = matthews_corrcoef(y_test, y_pred)

    logging.info(f"\n--- XGBoost Test Set Evaluation ---")
    logging.info(f"Test Accuracy: {test_accuracy*100:.2f}%, Test MCC: {test_mcc:.4f}")

    logging.info("\nClassification Report (sklearn):")
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    for line in report.split('\n'):
        logging.info(line)
        
    # 6. Save Model and Plots
    model.save_model(MODEL_FILENAME)
    logging.info(f"XGBoost model saved to '{MODEL_FILENAME}'")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='gain') # 'gain' is often a good metric
    plt.title('XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig("binary_xgb_feature_importance.png")
    plt.close()
    logging.info("Feature importance plot saved.")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Binary XGBoost Confusion Matrix')
    plt.savefig("binary_xgb_confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix plot saved.")

if __name__ == "__main__":
    try:
        import xgboost
        import seaborn
    except ImportError as e:
        print(f"Missing library: {e.name}. Please run 'pip install xgboost seaborn'")
    else:
        train_binary_xgb()