import joblib
import pandas as pd
import time

from shap import SHAP

print("loading model...")
start_time = time.time()
california_model = joblib.load('./california/california_model.pkl')
print(f"done loading model after {round(time.time() - start_time, 6)} seconds.")

print("loading test data...")
start_time = time.time()
test_data = pd.read_csv('./california/california_test_data.csv')
print(f"done loading test data after {round(time.time() - start_time, 6)} seconds.")

print("initializing SHAP")
start_time = time.time()
california_shap = SHAP(california_model, test_data.drop('target', axis=1), 0)
print(f"done initializing SHAP after {round(time.time() - start_time, 6)} seconds")

print("computing single SHAP values...")
start_time = time.time()
single_shap = california_shap.get_shap()
print(f"done computing single SHAP values after {round(time.time() - start_time, 6)} seconds")

print("computing pairwise SHAP values...")
start_time = time.time()
pairwise_shap = california_shap.get_shap_interactions()
print(f"done computing pairwise SHAP values after {round(time.time() - start_time, 6)} seconds")

print("RESULTS")
print("SHAP values:")
print(single_shap)

print("SHAP interaction values:")
print(pairwise_shap)