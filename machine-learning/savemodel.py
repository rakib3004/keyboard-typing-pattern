import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import joblib

data = pd.read_csv('pattern.csv')
X = data.drop(columns=['Target','user'])
y = data['Target']
for value in y.values:
    print(value)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

joblib.dump(model, 'typing_pattern.pkl')