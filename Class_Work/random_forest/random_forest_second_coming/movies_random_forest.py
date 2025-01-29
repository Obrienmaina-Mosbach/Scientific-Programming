import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../Scientific-Programming/data/movie_dataset.csv")
print(data.head())
data.info()
print(data.isnull().sum())



# Drop irrelevant string columns that can't be converted to numeric
data_cleaned = data.drop(columns=['homepage', 'original_title', 'title', 'tagline', 'overview', 'keywords', 'original_language', 'production_companies', 'country', 'spoken_languages', 'status', 'cast', 'crew', 'director'])

# Convert 'Date' column to datetime format
data_cleaned['release_date'] = pd.to_datetime(data_cleaned['release_date'])

# Extract the month from the 'Date' column
data_cleaned['Month'] = data_cleaned['release_date'].dt.to_period('M')  # Month-Year format (e.g., '2025-01')

# Fill or encode remaining non-numeric columns (like 'genres')
data_cleaned['genres'] = data_cleaned['genres'].fillna('Unknown')

# Perform one-hot encoding for 'genres'
genres_encoded = pd.get_dummies(data_cleaned['genres'], prefix='genre')
data_cleaned = pd.concat([data_cleaned.drop(columns=['genres']), genres_encoded], axis=1)

# Check that all columns are now numeric
print(data_cleaned.dtypes)

X = data_cleaned.drop(columns=['revenue'])
y = data_cleaned['revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# 1. Train the model
model = LinearRegression()  # Initialize the Linear Regression model
model.fit(X_train, y_train)  # Train the model using training data

# 2. Predict on the test set
y_pred = model.predict(X_test)

# 3. Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2 = r2_score(y_test, y_pred)            # Calculate R² Score

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)