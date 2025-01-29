import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

data = pd.read_csv("../Scientific-Programming/data/Sample_Superstore 2.csv", encoding='ISO-8859-1')

# Exploratory Data Analysis (EDA)
# Visualizing the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())
print(data.info())
print(data.describe())

# Encode categorical features
categorical_cols = ['Region', 'Category', 'Sub-Category']  # Update as per your dataset
encoder = LabelEncoder()

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Handle missing values if any
data = data.fillna('Unknown')

# Scale numerical features
scaler = StandardScaler()
data[['Sales', 'Profit', 'Quantity', 'Discount']] = scaler.fit_transform(
    data[['Sales', 'Profit', 'Quantity', 'Discount']]
)

# Split the dataset into features and target
X = data.drop(columns=['Sales', 'Profit', 'Order ID'])  # Features
y_sales = data['Sales']  # Target for sales prediction
y_profit = data['Profit']  # Target for profit prediction

# Train-test split
X_train, X_test, y_sales_train, y_sales_test = train_test_split(X, y_sales, test_size=0.2, random_state=42)
X_train, X_test, y_profit_train, y_profit_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)

print(data['Order Date'].dtype)
print(data['Order Date'].head())

# Check if there are any invalid dates (NaT values)
print(data['Order Date'].isna().sum())  # Count NaT (Not a Timestamp)

# Train a model
sales_model = RandomForestRegressor(n_estimators=100, random_state=42)
sales_model.fit(X_train, y_sales_train)