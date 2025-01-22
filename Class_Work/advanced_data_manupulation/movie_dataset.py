import pandas as pd

moviedata = pd.read_csv("../Scientific-Programming/data/movie_dataset.csv")
netflixdata = pd.read_csv("../Scientific-Programming/data/netflix_titles.csv")

dfm = pd.DataFrame(moviedata)
dfn = pd.DataFrame(netflixdata)

merged_df = pd.merge(dfm, dfn, on='title', how='left') # Intersection
print("Merged DataFrame:\n", merged_df.head(10))



#-----------------------------------------------------------------------------------------------


# Calculate profitability (revenue - budget)
merged_df["Profit"] = merged_df["revenue"] - merged_df["budget"]

# Group by country to find the most profitable countries
country_profitability = merged_df.groupby("title")["Profit"].sum().reset_index()

# Sort by profitability in descending order to find the most profitable countries
country_profitability = country_profitability.sort_values(by="Profit", ascending=False)

# Display the results
print(country_profitability.head(10))


#----------------------------------------------------------------------------------------------
# Rank the movies by 'Profit', in descending order (highest profit first)
merged_df["Profit_Rank"] = merged_df["Profit"].rank(ascending=False, method="min")

# Display the DataFrame with the new columns
print(merged_df[["title", "Profit", "Profit_Rank"]].head())  # Example showing title, profit, and rank


#-------------------------------------------------------------------------------------------------------
# If genres are in a list or need to be split, you may want to explode the genres into separate rows
# Assuming genres are in a single column, and multiple genres are separated by commas, split them:
merged_df["genres"] = merged_df["genres"].str.split(",")  # Split genres by comma if necessary

# Explode the genres list into separate rows, so each genre has its own row
merged_df_exploded = merged_df.explode("genres")

# Group by genre and sum the profits
genre_profit = merged_df_exploded.groupby("genres")["Profit"].sum().reset_index()

# Sort the genres by total profit in descending order
genre_profit_sorted = genre_profit.sort_values(by="Profit", ascending=False)

# Display the most profitable genres
print(genre_profit_sorted.head(10))