import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import xgboost as xgb
from xgboost import XGBClassifier

# 1. Load and prepare the dataset
data = pd.read_csv('your_dataset.csv')

# Split features and target (LPP)
X = data.drop(columns=['LPP'])
y = data['LPP']

# Split into train and test sets (holdout for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Impute missing values in feature columns using XGBoost
def impute_missing_values(X, y):
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            print(f"Imputing missing values for {col}...")
            # Split the data for training the imputation model
            X_impute_train = X[X[col].notnull()]
            X_impute_test = X[X[col].isnull()]
            y_impute_train = X_impute_train[col]
            
            # Train an XGBoost model to predict the missing values
            xgb_imputer = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            xgb_imputer.fit(X_impute_train.drop(columns=[col]), y_impute_train)
            
            # Predict and fill the missing values
            X.loc[X[col].isnull(), col] = xgb_imputer.predict(X_impute_test.drop(columns=[col]))
    return X

# Impute missing values in feature columns
X_train = impute_missing_values(X_train, y_train)
X_test = impute_missing_values(X_test, y_test)

# 3. Train the XGBoost model to predict the LPP column
def train_xgb_classifier(X_train, y_train, X_test, y_test, metric='accuracy'):
    xgb_model = XGBClassifier(objective='multi:softmax', n_estimators=100, random_state=42)
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Evaluate model
    if metric == 'accuracy':
        train_score = accuracy_score(y_train, y_pred_train)
        test_score = accuracy_score(y_test, y_pred_test)
    elif metric == 'f1':
        train_score = f1_score(y_train, y_pred_train, average='weighted')
        test_score = f1_score(y_test, y_pred_test, average='weighted')
    elif metric == 'log_loss':
        train_score = log_loss(y_train, xgb_model.predict_proba(X_train))
        test_score = log_loss(y_test, xgb_model.predict_proba(X_test))
    
    print(f"Train {metric.upper()}: {train_score}")
    print(f"Test {metric.upper()}: {test_score}")
    
    return xgb_model, train_score, test_score

# Perform multiple evaluations using different metrics
metrics = ['accuracy', 'f1', 'log_loss']
best_score = float('-inf')  # In classification, we want to maximize the score
best_metric = None
best_model = None

for metric in metrics:
    print(f"Evaluating with {metric.upper()} metric:")
    model, train_score, test_score = train_xgb_classifier(X_train, y_train, X_test, y_test, metric=metric)
    if test_score > best_score:  # Looking for maximum score
        best_score = test_score
        best_metric = metric
        best_model = model

print(f"\nBest model based on {best_metric.upper()} with test score: {best_score}")

# 4. Final Evaluation on holdout test set
y_test_pred = best_model.predict(X_test)

# Calculate final evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='weighted')
logloss = log_loss(y_test, best_model.predict_proba(X_test))

print("\nFinal Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"Log Loss: {logloss}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))