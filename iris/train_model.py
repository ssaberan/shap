import joblib
import pandas as pd
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load data (using Iris dataset as an example)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')

# Save train and test data
train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data['target'] = y_train
train_data.to_csv('train_data.csv', index=False)

test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
