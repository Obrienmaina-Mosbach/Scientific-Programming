import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("https://raw.githubusercontent.com/nileshely/SuperStore-Dataset-2019-2022/main/superstore_dataset.csv")


# Viewing data
print("First 10 rows:\n", data.head(), data.describe())
print("\nSummary Info:\n")
data.info()


'''# Check for missing values in each column
missing_values = data.isnull().sum()
print(missing_values)'''

'''# Check if any column has missing values
columns_with_missing = data.isnull().any()
print(columns_with_missing)'''

# Total missing values in the entire dataset
total_missing = data.isnull().sum().sum()
print(f"Total missing values: {total_missing}")

'''# Visualize missing values as a heatmap
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()


missing_values.plot(kind='bar', figsize=(10, 6))
plt.title("Missing Values Count per Column")
plt.xlabel("Columns")
plt.ylabel("Missing Values Count")
plt.show()'''

# Check if 'Revenue' column is missing, and if so, create it
if 'Revenue' not in data.columns:
    if 'quantity' in data.columns and 'sales' in data.columns:
        data['Revenue'] = data['quantity'] * (data['sales']/data['quantity'])
        print("Revenue column created successfully!")
    else:
        print("Error: The required columns 'Quantity' and 'Price' are missing.")
else:
    print("'Revenue' column already exists.")

# Display the first few rows of the DataFrame
print(data.head())


# Check if 'Date' column exists
if 'order_date' in data.columns:
    # Convert the 'Date' column to datetime format
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')

    # Check for any rows where conversion failed
    if data['order_date'].isnull().any():
        print("Some dates could not be converted. Check for invalid formats or missing values.")
    else:
        print("Date column successfully converted to datetime format!")
else:
    print("The 'Date' column is not present in the dataset.")

# Display the first few rows to verify the changes
print(data.head())

# Check if 'Revenue' or 'Sales' column exists in the dataset
if 'Revenue' in data.columns:
    total_revenue = data['Revenue'].sum()
    print(f"Total revenue generated: {total_revenue}")
elif 'sales' in data.columns:
    total_revenue = data['sales'].sum()
    print(f"Total revenue generated: {total_revenue}")
else:
    print("Error: No 'Revenue' or 'sales' column found in the dataset.")


# Ensure the 'Revenue' column is present
if 'Revenue' in data.columns and 'product_name' in data.columns:
    # Group by 'Product Name' and calculate total revenue for each product
    top_products = data.groupby('product_name')['Revenue'].sum().sort_values(ascending=False).head(5)
    
    print("Top 5 products by total revenue:")
    print(top_products)
elif 'sales' in data.columns and 'product_name' in data.columns:
    # If 'Revenue' is not available, use 'Sales' instead
    top_products = data.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(5)
    
    print("Top 5 products by total sales:")
    print(top_products)
else:
    print("Error: Required columns ('Revenue' or 'sales' and 'product_name') are missing.")

# Check if 'Customer Name' column exists
if 'customer' in data.columns:
    # Group by 'Customer Name' and count the number of orders
    most_orders = data.groupby('customer').size().sort_values(ascending=False).head(1)
    
    print("Customer who placed the most orders:")
    print(most_orders)
else:
    print("Error: The dataset does not have a 'Customer Name' column.")

# Check if 'Product Name' exists
if 'product_name' in data.columns:
    # Group by 'Product Name' and calculate aggregations
    grouped_data = data.groupby('product_name').agg({
        'Revenue': 'sum',       # Total revenue per product
        'sales': 'sum',         # Total sales (if 'Revenue' is missing)
        'quantity': 'sum',      # Total quantity sold per product
        'order_id': 'count'     # Number of orders for each product
    }).reset_index()

    # Rename columns for clarity
    grouped_data.rename(columns={'Order ID': 'Order Count'}, inplace=True)

    print("Grouped data by Product Name:")
    print(grouped_data.head())  # Display the first few rows of the grouped data
else:
    print("Error: The dataset does not have a 'Product Name' column.")

# Check if 'Product Name' and 'Quantity' columns exist
if 'product_name' in data.columns and 'quantity' in data.columns:
    # Group by 'Product Name' and calculate total quantity sold
    total_quantity_per_product = data.groupby('product_name')['quantity'].sum().sort_values(ascending=False)
    
    print("Total quantity sold for each product:")
    print(total_quantity_per_product)
else:
    print("Error: Required columns 'Product Name' or 'Quantity' are missing.")

# Ensure the 'Date' column is in datetime format
if 'order_date' in data.columns:
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
    
    # Check for 'Revenue' column
    if 'Revenue' in data.columns:
        # Group by 'Order Date' and calculate total daily revenue
        daily_revenue = data.groupby('order_date')['Revenue'].sum().reset_index()
        daily_revenue.rename(columns={'order_date': 'Date', 'Revenue': 'Daily Revenue'}, inplace=True)
        
        print("Daily revenue trends:")
        print(daily_revenue.head())  # Display the first few rows
    else:
        print("Error: 'Revenue' column is missing from the dataset.")
else:
    print("Error: 'Order Date' column is missing or not formatted correctly.")


import matplotlib.pyplot as plt

# Check if necessary columns exist
if 'product_name' in data.columns and 'Revenue' in data.columns:
    # Group by 'Product Name' and calculate total revenue
    top_products = data.groupby('product_name')['Revenue'].sum().sort_values(ascending=False).head(5)
    
    # Plot a bar chart
    plt.figure(figsize=(10, 6))
    top_products.plot(kind='bar', color='skyblue')
    plt.title('Top 5 Products by Revenue', fontsize=16)
    plt.xlabel('Product Name', fontsize=12)
    plt.ylabel('Total Revenue', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Error: Required columns ('Product Name' and 'Revenue') are missing.")


# Ensure 'Order Date' is in datetime format
if 'order_date' in data.columns and 'Revenue' in data.columns:
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
    
    # Group by 'Order Date' and calculate daily revenue
    daily_revenue = data.groupby('order_date')['Revenue'].sum().reset_index()
    
    # Plot a line graph
    plt.figure(figsize=(12, 6))
    plt.plot(daily_revenue['order_date'], daily_revenue['Revenue'], color='blue', marker='o', linestyle='-', label='Daily Revenue')
    plt.title('Daily Revenue Trends', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Revenue', fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()
else:
    print("Error: Required columns ('Order Date' and 'Revenue') are missing.")


# Ensure 'Order Date' is in datetime format and calculate daily revenue
if 'order_date' in data.columns and 'Revenue' in data.columns:
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
    
    # Group by 'Order Date' and calculate daily revenue
    daily_revenue = data.groupby('order_date')['Revenue'].sum().reset_index()
    
    # Sort by date to ensure correct order for rolling calculation
    daily_revenue = daily_revenue.sort_values('order_date')
    
    # Calculate the 7-day moving average
    daily_revenue['7-Day Moving Average'] = daily_revenue['Revenue'].rolling(window=7).mean()
    
    print(daily_revenue.head(10))  # Display the first 10 rows
else:
    print("Error: Required columns ('Order Date' and 'Revenue') are missing.")


# Ensure 'Order Date' and 'Revenue' are in the dataset
if 'order_date' in data.columns and 'Revenue' in data.columns:
    # Ensure 'Order Date' is in datetime format
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
    
    # Group by 'Order Date' to calculate daily revenue
    daily_revenue = data.groupby('order_date')['Revenue'].sum().reset_index()
    
    # Identify the day with the highest revenue
    highest_revenue_day = daily_revenue.loc[daily_revenue['Revenue'].idxmax()]
    highest_revenue_date = highest_revenue_day['order_date']
    print(f"Day with the highest revenue: {highest_revenue_date}, Revenue: {highest_revenue_day['Revenue']}")
    
    # Filter the original dataset for that day
    data_on_highest_revenue_day = data[data['order_date'] == highest_revenue_date]
    
    # Identify the product with the highest revenue contribution on that day
    if 'product_name' in data.columns:
        top_product = data_on_highest_revenue_day.groupby('product_name')['Revenue'].sum().sort_values(ascending=False).head(1)
        print("Product contributing the most revenue:")
        print(top_product)
    else:
        print("Error: 'Product Name' column is missing in the dataset.")
else:
    print("Error: Required columns ('Order Date' and 'Revenue') are missing.")
