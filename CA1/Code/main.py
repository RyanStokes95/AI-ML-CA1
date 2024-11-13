import pandas as pd

# Path to dataset
dataset_path = r'CA1\\Datasets\Cardiovascular\cardio_train.csv'

# Load the dataset into a pandas
df = pd.read_csv(dataset_path, sep=';')

# Changing age in days to age in years
df['age_years'] = (df['age'] / 365.25).astype(int)
df['age_years'] = df['age_years'].astype('int64')
df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
print(df[['age', 'age_years']].head())
df.drop(columns=['age'], inplace=True)
cols = list(df.columns)
cols.insert(2, cols.pop(cols.index('age_years')))
df = df[cols]

#Check for missing data
print(df.isnull().sum())

#Setting dtype
binary_columns = ['gender', 'smoke', 'alco', 'active', 'cardio']
df[binary_columns] = df[binary_columns].astype('category')

numeric_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Eliminating bad data (Negative blood pressure values and unrealistic weigh and height for an adult)
df = df[(df['ap_hi'] > 20) & (df['ap_hi'] < 300)]
df = df[(df['ap_lo'] > 20) & (df['ap_lo'] < 300)]
df = df[(df['height'] > 50) & (df['height'] < 250)]
df = df[(df['weight'] > 30) & (df['weight'] < 250)]

print(df.dtypes)

pd.set_option('display.max_columns', None)
print(df.describe(include='all'))
