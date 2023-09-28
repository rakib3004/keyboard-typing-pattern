import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

single_data_point = {'H.period': 0.065, 'DD.period.t': 0.185, 'UD.period.t': 0.12, 'H.t': 0.115, 'DD.t.i': 0.211, 'UD.t.i': 0.096, 'H.i': 0.09, 'DD.i.e': 0.225, 'UD.i.e': 0.135, 'H.e': 0.139, 'DD.e.5': 0.321, 'UD.e.5': 0.182, 'H.5': 0.104, 'DD.5.R': 0.449, 'UD.5.R': 0.345, 'H.R': 0.098, 'DD.R.o': 0.491, 'UD.R.o': 0.393, 'H.o': 0.085, 'DD.o.a': 0.157, 'UD.o.a': 0.072, 'H.a': 0.174, 'DD.a.n': 0.182, 'UD.a.n': 0.008, 'H.n': 0.081, 'DD.n.l': 0.293, 'UD.n.l': 0.212, 'H.l': 0.08, 'DD.l.enter': 0.327, 'UD.l.enter': 0.247, 'H.enter': 0.084}
# Create a DataFrame from the single data point
single_data_df = pd.DataFrame([single_data_point])

data = pd.read_csv('finaldata.csv')

X = data.drop(columns=['Target','user'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest Classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Make a prediction for the single data point
single_data_pred_prob = model.predict_proba(single_data_df)[:, 1]  # Probability of being the positive class

# Set the threshold (0.5 in this case) to classify the prediction
threshold = 0.5
prediction = (single_data_pred_prob > threshold).astype(int)

# Check the prediction
if prediction == 1:
    print("The prediction is above the threshold (greater than 0.5).")
else:
    print("The prediction is below the threshold (less than or equal to 0.5).")
