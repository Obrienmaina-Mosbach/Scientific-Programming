import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({'key': [1, 2, 3], 'A': ['a1', 'a2', 'a3']})
df2 = pd.DataFrame({'key': [2, 3, 4], 'B': ['b1', 'b2', 'b3']})

# Merge the DataFrames on the 'key' column
merged_df = pd.merge(df1, df2, on='key', how='inner') # Intersection
print("Merged DataFrame:\n", merged_df)

# Concatenate
df_concat = pd.concat([df1, df2], axis=0)  # Stack rows
print("\nConcatenated DataFrame:\n", df_concat)


data = {'Region': ['North', 'South', 'North', 'West'], 'Sales': [200, 150, 300, 400], 'Profit': [20, 10, 40, 50]}
df = pd.DataFrame(data)
# Grouping and Aggregation 
grouped = df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'mean'}) 
print("Grouped and Aggregated Data:\n", grouped)
