#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Path to dataset
dataset_path = r'CA1\\Datasets\Cardiovascular\cardio_train.csv'

# Load the dataset into pandas
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

# Check for missing data
print(df.isnull().sum())

# Setting datatype
binary_columns = ['gender', 'smoke', 'alco', 'active', 'cardio']
df[binary_columns] = df[binary_columns].astype('category')

numeric_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Eliminating bad data (Negative blood pressure values and unrealistic weigh and height for an adult)
df = df[(df['ap_hi'] > 20) & (df['ap_hi'] < 300)]
df = df[(df['ap_lo'] > 20) & (df['ap_lo'] < 300)]
df = df[(df['height'] > 120) & (df['height'] < 250)]
df = df[(df['weight'] > 40) & (df['weight'] < 250)]

print(df.dtypes)

pd.set_option('display.max_columns', None)
print(df.describe(include='all'))

#Visualisations
# Scatter plot for Height vs. Weight with smaller dots
plt.figure(figsize=(10, 8))

# Set the size and colour of the dots
plt.scatter(df['height'], df['weight'], alpha=0.5, color='blue', s=5)

# Add labels and title
plt.title('Scatter Plot: Height/Weight', fontsize=16)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Weight (kg)', fontsize=12)

# Show plot
plt.show()

# Scatter plot for Blood Pressure
plt.figure(figsize=(10, 8))

# Set the size and colour of the dots
plt.scatter(df['ap_hi'], df['ap_lo'], alpha=0.5, color='red', s=5)

# Add labels and title
plt.title('Scatter Plot: Blood Pressure', fontsize=16)
plt.xlabel('Systolic blood pressure (mmHg)', fontsize=12)
plt.ylabel('Diastolic blood pressure (mmHg)', fontsize=12)

# Show plot
plt.show()

#Scaling using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']])
scaled_df = pd.DataFrame(scaled_features, columns=['age_years', 'height', 'weight', 'ap_hi', 'ap_lo'], index=df.index)
df_scaled = pd.concat([scaled_df, df[['cholesterol', 'gluc', 'gender', 'smoke', 'alco', 'active', 'cardio']]], axis=1)


#Model Implementation
# Split data into features (X) and target (y)
X = df_scaled.drop(columns=['cardio'])
y = df_scaled['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=295)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report KNN:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evalate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report LR:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()