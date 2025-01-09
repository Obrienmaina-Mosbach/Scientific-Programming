import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the weather dataset (replace 'weather_data.csv' with your actual file path)
df = pd.read_csv('../Scientific-Programming/data/weather_data.csv')



# Print the column names to inspect
print(df.columns)

# Convert the 'date' column to datetime
df['Date_Time'] = pd.to_datetime(df['Date_Time'])

# Extract the month and year for grouping
df['month'] = df['Date_Time'].dt.month
df['year'] = df['Date_Time'].dt.year

# Group by year and month to get the average temperature for each month
monthly_temp = df.groupby(['year', 'month'])['Temperature_C'].mean().reset_index()

# Plot the monthly temperature trends using sns.lineplot()
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_temp, x='month', y='Temperature_C', hue='year', marker='o')

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.title('Monthly Temperature Trends')

# Show the plot
plt.show()

# Scatter plot to show the relationship between humidity and temperature
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Humidity_pct', y='Temperature_C', alpha=0.7, color='green')

# Set labels and title
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.title('Relationship Between Humidity and Temperature')

# Show the plot
plt.show()


