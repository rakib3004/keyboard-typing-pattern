# Single data point as a dictionary (replace this with your actual data)
single_data_point = {
    'H.period': 0.1,
    'DD.period.t': 0.2,
    'UD.period.t': 0,
    'H.t': 0.15,
    # Add other features here...
}

# Create a DataFrame from the single data point
single_data_df = pd.DataFrame([single_data_point])

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
