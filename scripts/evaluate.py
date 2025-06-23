# scripts/evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score
import json

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Load the test data
print("Loading test data for evaluation...")
test_df = pd.read_csv(config['test_data_path'])
X_test = test_df.drop('Exited', axis=1)
y_test = test_df['Exited']

# Load the trained model
print("Loading the trained model...")
model = joblib.load(config['model_path'])

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test)

# Calculate metrics
f1 = f1_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy}")
print(f"Model F1 Score: {f1}")

# Save metrics to a file
metrics = {
    'accuracy': accuracy,
    'f1_score': f1
}
with open(config['metric_path'], 'w') as f:
    json.dump(metrics, f)

print(f"Metrics saved to {config['metric_path']}")