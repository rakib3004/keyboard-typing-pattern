import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'registration.csv'

# Read the CSV file into a DataFrame
dataset = pd.read_csv(file_path)
collection = dataset.drop(columns=['user'])
data = collection.drop(columns=['Target'])


# Get the column names (titles)
column_names = data.columns

# Create a dictionary to store the pairs of column names and their first values
single_data_point = {}

# Iterate through the column names
for column_name in column_names:
    # Get the first value from the column (iloc[0] selects the first row)
    first_value = data[column_name].iloc[3]
    
    # Add the pair to the dictionary
    single_data_point[column_name] = first_value

# Print the resulting dictionary
print(single_data_point)
