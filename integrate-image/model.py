import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import joblib
import sys

if __name__ == "__main__":
    if len(sys.argv) >= 2:
            user_input_param = sys.argv[1]
            user_input = json.loads(user_input_param)

            data = pd.read_csv('pattern.csv')
            X = data.drop(columns=['Target','user'])
            y = data['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            user_data = pd.DataFrame([user_input])

            prediction = model.predict(user_data)

            result  = int(prediction[0])
            print(json.dumps({"result": result}))

