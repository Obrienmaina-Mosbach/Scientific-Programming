import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

from Future_Of_Jobs_Project.future_jobs import y_test

# Load job postings dataset
jobs_df = pd.read_csv('./data/postings.csv')

# Convert original_listed_time from milliseconds to datetime format
jobs_df['original_listed_time'] = pd.to_datetime(jobs_df['original_listed_time'], unit='ms')

# Extract year and month
jobs_df['year'] = jobs_df['original_listed_time'].dt.year
jobs_df['month'] = jobs_df['original_listed_time'].dt.month

# Aggregate job postings by job title, job skills, and location
job_trends = jobs_df.groupby(['title', 'location', 'year', 'month']).size().reset_index(name='job_count')

# Sort by time
job_trends = job_trends.sort_values(by=['title', 'location', 'year', 'month'])

# Shift job count to use the previous month's demand as a feature
job_trends['job_count_last_month'] = job_trends.groupby(['title', 'location'])['job_count'].shift(1)

# Rolling average of job demand for the past 3 months
job_trends['job_count_avg_3m'] = job_trends.groupby(['title', 'location'])['job_count'].rolling(3).mean().reset_index(drop=True)

# Handle missing values by making them 0
job_trends['job_count_last_month'] = job_trends['job_count_last_month'].fillna(0)
job_trends['job_count_avg_3m'] = job_trends['job_count_avg_3m'].fillna(0)

# Select features & target
features = ['year', 'month', 'title', 'location', 'job_count_last_month', 'job_count_avg_3m']
X = job_trends[features]
y = job_trends['job_count']

# Train ARIMA model
arima_order = (5, 1, 0)  # Example order, you may need to tune this
arima_model = ARIMA(y, order=arima_order)
arima_result = arima_model.fit()

# Predictions
y_pred_arima = arima_result.predict(start=len(y) - len(y_test), end=len(y) - 1, dynamic=False)

# Model Evaluation Metrics
mse_arima = mean_squared_error(y_test, y_pred_arima)
r2_arima = r2_score(y_test, y_pred_arima)
print(f'ARIMA MSE: {mse_arima}, R^2: {r2_arima}')

# Time Series Plot
plt.figure(figsize=(12, 6))
plt.plot(job_trends['original_listed_time'], job_trends['job_count'], label='Actual')
plt.plot(job_trends['original_listed_time'].iloc[-len(y_test):], y_pred_arima, label='Predicted', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Job Count')
plt.title('Job Posting Trends Over Time')
plt.legend()
plt.show()

# Residual Plot
residuals = y_test - y_pred_arima
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_arima, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Job Count')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Histogram of Errors
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()

# Pairplot of Features
sns.pairplot(job_trends[features])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(job_trends[features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Actual vs Predicted Job Counts
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_arima, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Job Count')
plt.ylabel('Predicted Job Count')
plt.title('Actual vs Predicted Job Counts (ARIMA)')
plt.show()