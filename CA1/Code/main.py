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

# Define the custom bins and labels for age ranges
bins = [20, 30, 40, 50, 60, 70]
labels = ['21-30', '31-40', '41-50', '51-60', '61-70']

# Create a new column 'age_group' based on the bins
df['age_group'] = pd.cut(df['age_years'], bins=bins, labels=labels, right=False)

# Plot the histogram
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='age_group', data=df, palette='Set2')

# Shows count for first range with count < 10
for p in ax.patches:
    count = p.get_height()
    if count < 10:
        ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., count), 
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

#Plots Histogram
plt.title('Age Distribution by Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)

# Shows Histogram
plt.show()

# Scatter plot for Height vs. Weight with smaller dots
plt.figure(figsize=(20, 12))

# Set the size and colour of the dots
plt.scatter(df['height'], df['weight'], alpha=0.5, color='blue', s=5)

# Add labels and title
plt.title('Scatter Plot: Height/Weight', fontsize=16)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Weight (kg)', fontsize=12)

# Show plot
plt.show()

# Scatter plot for Blood Pressure
plt.figure(figsize=(20, 12))

# Set the size and colour of the dots
plt.scatter(df['ap_hi'], df['ap_lo'], alpha=0.5, color='red', s=5)

# Add labels and title
plt.title('Scatter Plot: Blood Pressure', fontsize=16)
plt.xlabel('Systolic blood pressure (mmHg)', fontsize=12)
plt.ylabel('Diastolic blood pressure (mmHg)', fontsize=12)

# Show plot
plt.show()

sns.countplot(x='cardio', data=df)
plt.title('Distribution of Cardio (0 vs 1)')
plt.show()
sns.countplot(x='smoke', data=df)
plt.title('Distribution of Smoking (0 vs 1)')
plt.show()
sns.countplot(x='alco', data=df)
plt.title('Distribution of Alcohol Consumption (0 vs 1)')
plt.show()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']])
scaled_df = pd.DataFrame(scaled_features, columns=['age_years', 'height', 'weight', 'ap_hi', 'ap_lo'], index=df.index)

# Combine scaled features with the other columns
df_scaled = pd.concat([scaled_df, df[['cholesterol', 'gluc', 'gender', 'smoke', 'alco', 'active', 'cardio']]], axis=1)

print(df_scaled.isnull().sum())

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
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()