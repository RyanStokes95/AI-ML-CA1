import pandas as pd

# Specify the path to your dataset (update with the correct path)
dataset_path = r'CA1\\Datasets\Cardiovascular\cardio_train.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(dataset_path, sep=';')

# Display the first few rows of the dataset to confirm it's loaded correctly
print(df.head())