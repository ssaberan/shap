import joblib
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost regression model
model = xgb.XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'california_model.pkl')

# Save train and test data
train_data = pd.DataFrame(X_train, columns=california.feature_names)
train_data['target'] = y_train
train_data.to_csv('california_train_data.csv', index=False)

test_data = pd.DataFrame(X_test, columns=california.feature_names)
test_data['target'] = y_test
test_data.to_csv('california_test_data.csv', index=False)
