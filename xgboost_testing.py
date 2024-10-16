import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the dataset
# Assuming 'dataset.csv' is the file containing the data
df = pd.read_csv('dataset.csv')

# Separate the feature columns (all columns except 'LPP')
X = df.drop(columns=['LPP'])

# Store LPP in a separate variable
y = df['LPP']

# Handle missing values in feature columns (filling missing values with the mean here)
X = X.fillna(X.mean())

# Split the data into train and test sets
# Only use rows where 'LPP' is not NaN for training
X_train = X[y.notna()]
y_train = y[y.notna()]

# We will later predict on rows where 'LPP' is NaN
X_test = X[y.isna()]

# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train the model on the non-missing LPP data
xgb_model.fit(X_train, y_train)

# Predict missing LPP values
y_pred = xgb_model.predict(X_test)

# Fill the missing values of LPP with the predicted values
df.loc[df['LPP'].isna(), 'LPP'] = y_pred

# Save the updated dataset with filled LPP values
df.to_csv('updated_dataset.csv', index=False)

print("Missing values of LPP have been predicted and saved to 'updated_dataset.csv'.")
