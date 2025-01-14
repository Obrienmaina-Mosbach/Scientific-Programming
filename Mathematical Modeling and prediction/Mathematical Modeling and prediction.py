import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

moviesdata = pd.read_csv('../Scientific-Programming/data/movie_dataset.csv')

# Inspect the dataset
print(moviesdata.head())

# Data Preprocessing
# Check for missing values
print(moviesdata.isnull().sum())

X_cleaned = moviesdata.dropna()  # Drops rows with any NaN values

# Encode categorical variables using one-hot encoding
moviesdata_encoded = pd.get_dummies(X_cleaned, drop_first=True)

#Replace missing data
#print('-------------Replacing Missing Data------------')
#moviesdata['genres'] = moviesdata['genres'].fillna('Unknown')
#moviesdata['homepage'] = moviesdata['homepage'].fillna('Unknown')
#moviesdata['keywords'] = moviesdata['keywords'].fillna('Unknown')
#moviesdata['overview'] = moviesdata['overview'].fillna('Unknown')
#moviesdata['release_date'] = moviesdata['release_date'].fillna('Unknown')
#moviesdata['runtime'] = moviesdata['runtime'].fillna('Unknown')
#moviesdata['tagline'] = moviesdata['tagline'].fillna('Unknown')
#moviesdata['cast'] = moviesdata['cast'].fillna('Unknown')
#moviesdata['director'] = moviesdata['director'].fillna('Unknown')
#print(moviesdata.isnull().sum())

# Select target variable (Budget) and features
x = moviesdata_encoded.drop(columns=['budget'])
y = moviesdata_encoded['budget']


print('-------------Split the data (80:20)%------------')
# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('-------------Time to Predict------------')
# Predict using Linear Regression model
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)  # MSE, no squared argument needed
#rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)  # RMSE, use squared=False
print(f'Linear Regression MAE: {mae_lr}')
print(f'Linear Regression MSE: {mse_lr}')
#print(f'Linear Regression RMSE: {rmse_lr}')


# Model 2: Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict using Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree model
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)  # MSE, no squared argument needed

#rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)  # RMSE, use squared=False
print(f'Decision Tree MAE: {mae_dt}')
print(f'Decision Tree MSE: {mse_dt}')
#print(f'Decision Tree RMSE: {rmse_dt}'

# Visualization: Comparison of Actual vs Predicted
# Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pred_dt, marker='o', linestyle='-', color='#005634', alpha=0.7)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()