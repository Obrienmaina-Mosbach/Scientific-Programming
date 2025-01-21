import pandas as pd

superdata = pd.read_csv("../Scientific-Programming/data/Sample_Superstore 2.csv", encoding='ISO-8859-1')


df = pd.DataFrame(superdata)

# Convert 'Date' column to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract the month from the 'Date' column
df['Month'] = df['Order Date'].dt.to_period('M')  # Month-Year format (e.g., '2025-01')

# Create the pivot table
pivot = df.pivot_table(values='Sales', index='Month', aggfunc='sum')

# Display the pivot table
print("Monthly Sales Trends Pivot Table:\n", pivot)

df['Sales'] = df['Sales'].fillna(0) #Replace missing sales with 0
df['Region'] = df['Region'].fillna('Unknown')
df['Category'] = df['Category'].fillna('Unknown')

print("DataFrame after handling missing values:\n", df)