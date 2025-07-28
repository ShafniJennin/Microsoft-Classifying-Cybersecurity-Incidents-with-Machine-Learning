import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

def train_model(X, y):
    model = XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    return {
        'f1_score': f1_score(y_val, preds, average='macro'),
        'precision': precision_score(y_val, preds, average='macro'),
        'recall': recall_score(y_val, preds, average='macro')
    }

def save_model(model, path='models/best_model.pkl'):
    joblib.dump(model, path)
