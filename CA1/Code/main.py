import pandas as pd

# Path to dataset
dataset_path = r'CA1\\Datasets\Cardiovascular\cardio_train.csv'

# Load the dataset into a pandas
df = pd.read_csv(dataset_path, sep=';')

# Test
print(df.head())