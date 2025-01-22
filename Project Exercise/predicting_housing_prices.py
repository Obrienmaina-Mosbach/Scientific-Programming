import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

housingdata = pd.read_csv("../Scientific-Programming/data/housing.csv")

# Display the first 10 rows of the dataset
print(housingdata.head(10))


#Check for missing values
print(housingdata.isnull().sum())

# Fill or drop missing values
housingdata = housingdata.fillna(0)

# Encode categorical variables
#Encoding categorical variables involves converting them into numerical representations that machine learning models can use
housingdata = pd.get_dummies(housingdata, drop_first=True)
print('-------------------')
#Check for missing values
print(housingdata.isnull().sum())

# Check feature correlation
plt.figure(figsize=(12, 8))
sns.heatmap(housingdata.corr(), annot=True, cmap="coolwarm")
plt.show()

print('-------------Split Data---------------')

# Define features (X) and target (y)
x = housingdata.drop("median_house_value", axis=1)  
y = housingdata["median_house_value"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('-------------Scaling the data---------------')
#Feature scaling is the process of normalizing or standardizing the range of features in a dataset so that they contribute equally to the model's learning process. 
#It ensures that features with larger numerical ranges do not dominate those with smaller ranges

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('-------------Train the Model---------------')
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, c=y_pred_rf, cmap='coolwarm')  # Color points based on predicted values -------------or----------- c='#FF5733' Using a hex color code
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.colorbar()  # Adds a color bar to the plot for reference
plt.show()



