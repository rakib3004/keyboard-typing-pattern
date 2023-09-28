from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Replace this import with the import of your actual trained model
# Import your model here, e.g., from your_module import your_model

# Replace this function with your actual prediction logic using the model
def make_prediction(single_data_point):
    # Example: Replace this with your model.predict() call
    # prediction = your_model.predict(input_data)

    # Load your dataset
    data = pd.read_csv('your_dataset.csv')

    # Split the dataset into features (X) and target labels (y)
    X = data.drop(columns=['Target'])
    y = data['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a machine learning model (Random Forest Classifier in this example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of being the positive class

    # Set a threshold (0.5 in this case) to classify predictions
    threshold = 0.5
    y_pred = (y_pred_prob > threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # You can now use this trained model to recognize keyboard typing patterns for any person.

    single_data_df = pd.DataFrame([single_data_point])

    # Make a prediction for the single data point
    single_data_pred_prob = model.predict_proba(single_data_df)[:, 1]  # Probability of being the positive class

    # Set the threshold (0.5 in this case) to classify the prediction
    threshold = 0.5
    prediction = (single_data_pred_prob > threshold).astype(int)
    return prediction

@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure that the expected keys are in the input data
        expected_keys = ['H.period', 'DD.period.t', 'UD.period.t', 'H.t']  # Add other keys as needed
        for key in expected_keys:
            if key not in data:
                return jsonify({'error': f'Missing key: {key}'})

        # Make predictions using your model
        prediction = make_prediction(data)

        return jsonify({'result': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
