import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import sys

if __name__ == "__main__":
    # Get the JSON string from the command-line arguments
    if len(sys.argv) >= 2:
            user_input_param = sys.argv[1]
            user_input = json.loads(user_input_param)

            model = joblib.load('typing_pattern.pkl')

            user_data = pd.DataFrame([user_input])

            prediction = model.predict(user_data)

            result  = int(prediction[0])
            print(json.dumps({"result": result}))
