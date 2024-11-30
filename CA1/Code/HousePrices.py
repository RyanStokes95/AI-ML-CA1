#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Path to dataset
dataset_path = r'CA1\\Datasets\House Prices\housingdata.csv'

# Load the dataset into pandas
df = pd.read_csv(dataset_path, sep=',')
#Exclude prices over 500,000 (Data capped at 501,000)
df = df[df['median_house_value'] <= 500000]

# Dropping columns with NaN values in total bedrooms
df.dropna(subset=['total_bedrooms'], inplace=True)

# Initializez the LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the 'ocean_proximity' column to numeric values (1,2,3)
df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])

#Visualisations
# Plot histogram for median_house_value with custom bins
plt.figure(figsize=(10, 6))
sns.histplot(df['median_house_value'], bins=30, kde=True, color='skyblue')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Histogram of Median House Values')
plt.show()

#Scaling and model training
# Splitting features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Results
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)


