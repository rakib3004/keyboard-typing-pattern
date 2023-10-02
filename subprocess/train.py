import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
# Load your dataset
data = pd.read_csv('finaldata.csv')
# Split the dataset into features (X) and target labels (y)
X = data.drop(columns=['Target','user'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest Classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of being the positive class
#print('y_pred_prob: ',y_pred_prob)
# Set a threshold (0.5 in this case) to classify predictions
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy * 100:.2f}%')
print(json.dumps({"accuracy": accuracy}))
# You can now use this trained model to recognize keyboard typing patterns for any person.
