import pandas as pd
import joblib
import re
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)

# ------------------ Preprocessing ------------------

def preprocess_text(text: str, phishing_keywords: Optional[List[str]] = None) -> str:
    """
    Preprocess email text: lowercase, remove most punctuation, keep URLs, append phishing keywords if present.
    Args:
        text (str): The email text.
        phishing_keywords (Optional[List[str]]): List of keywords to append if present.
    Returns:
        str: Preprocessed text.
    """
    try:
        text = text.lower()
        url_pattern = r'(https?://\S+|www\.\S+|\S+\.com)'
        urls = re.findall(url_pattern, text)
        
        # Remove punctuation but keep .com, http, etc.
        text = re.sub(r'[^a-zA-Z0-9\s:.\/_-]', '', text)

        if phishing_keywords is None:
            phishing_keywords = ['click', 'verify', 'login', 'urgent', 'account', 'download', 'payment']
        text += ' ' + ' '.join([word for word in phishing_keywords if word in text])
        return text
    except Exception as e:
        logging.error(f"Error in preprocess_text: {e}")
        return ""

# ------------------ Modular Functions ------------------

def load_and_prepare_data(filepath: str, phishing_keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and preprocess email data from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
        phishing_keywords (Optional[List[str]]): Custom phishing keywords for preprocessing.
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        if 'class' in df.columns:
            df = df[df['class'].astype(str).str.lower() != 'class']
            df = df.dropna(subset=['class'])
            df['class'] = df['class'].astype(int)
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain 'text' column.")
        df['text'] = df['text'].astype(str).apply(lambda x: preprocess_text(x, phishing_keywords))
        return df
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise

def train_model(
    df: pd.DataFrame,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any]:
    """
    Train a RandomForest model and TF-IDF vectorizer.
    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'class'.
        vectorizer_params (Optional[dict]): Parameters for TfidfVectorizer.
        model_params (Optional[dict]): Parameters for RandomForestClassifier.
    Returns:
        model, vectorizer
    """
    try:
        if 'class' not in df.columns:
            raise ValueError("DataFrame must contain 'class' column for training.")
        if vectorizer_params is None:
            vectorizer_params = {'ngram_range': (1, 2), 'max_features': 5000}
        if model_params is None:
            model_params = {'n_estimators': 100, 'random_state': 42}
        vectorizer = TfidfVectorizer(**vectorizer_params)
        X = vectorizer.fit_transform(df['text'])
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return clf, vectorizer
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise

def predict_emails(model, vectorizer, email_list: List[str], phishing_keywords: Optional[List[str]] = None):
    """
    Predict classes and confidences for a list of emails.
    Args:
        model: Trained model.
        vectorizer: Trained vectorizer.
        email_list (List[str]): List of email texts.
        phishing_keywords (Optional[List[str]]): Custom phishing keywords for preprocessing.
    Returns:
        predictions, confidences
    """
    try:
        processed = [preprocess_text(email, phishing_keywords) for email in email_list]
        X = vectorizer.transform(processed)
        predictions = model.predict(X)
        if hasattr(model, 'predict_proba'):
            confidences = model.predict_proba(X)
        else:
            confidences = [[1.0 if pred == 1 else 0.0, 1.0 if pred == 0 else 0.0] for pred in predictions]
        return predictions, confidences
    except Exception as e:
        logging.error(f"Error in predict_emails: {e}")
        raise

def evaluate_model(model, vectorizer, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate the model on a labeled DataFrame.
    Args:
        model: Trained model.
        vectorizer: Trained vectorizer.
        df (pd.DataFrame): DataFrame with 'text' and 'class'.
    Returns:
        dict: Metrics (accuracy, precision, recall, f1, confusion matrix, classification report)
    """
    try:
        X = vectorizer.transform(df['text'])
        y_true = df['class']
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=1),
            'recall': recall_score(y_true, y_pred, zero_division=1),
            'f1': f1_score(y_true, y_pred, zero_division=1),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=1)
        }
        return metrics
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        raise

def save_model(model, vectorizer, model_path: str, vectorizer_path: str):
    """
    Save the model and vectorizer to disk.
    """
    try:
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        logging.info(f"Model saved to {model_path}, vectorizer saved to {vectorizer_path}")
    except Exception as e:
        logging.error(f"Error saving model/vectorizer: {e}")
        raise

def load_model(model_path: str, vectorizer_path: str):
    """
    Load the model and vectorizer from disk.
    """
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logging.info(f"Model loaded from {model_path}, vectorizer loaded from {vectorizer_path}")
        return model, vectorizer
    except Exception as e:
        logging.error(f"Error loading model/vectorizer: {e}")
        raise

# ------------------ Load Dataset ------------------

df = pd.read_csv("data/custom_emails.csv")
df.columns = df.columns.str.strip().str.lower()
# Remove any rows where 'class' is literally 'class' or NaN
df = df[df['class'].astype(str).str.lower() != 'class']
df = df.dropna(subset=['class'])

# Now safely convert to int
df['class'] = df['class'].astype(int)
print("Loaded columns:", df.columns.tolist())
print(df.head())

# Check if necessary columns exist
if 'text' not in df.columns or 'class' not in df.columns:
    raise ValueError("CSV file must contain 'text' and 'class' columns.")

# ------------------ Preprocess ------------------

df['text'] = df['text'].astype(str).apply(preprocess_text)

print("\nClass distribution:\n", df['class'].value_counts())

# ------------------ Vectorization ------------------

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['class']

# ------------------ Train-Test Split ------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ------------------ Model Training ------------------

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ------------------ Evaluation ------------------

y_pred = clf.predict(X_test)
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=1))

# ------------------ Save Model and Vectorizer ------------------

joblib.dump(clf, "model/rf_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("\n✅ Model and vectorizer saved!")