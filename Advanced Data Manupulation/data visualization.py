import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths (adjust according to your folder structure)
ratings_path = '../Scientific-Programming/data/ml-100k/u.data'  # Ratings data
movies_path = '../Scientific-Programming/data/ml-100k/u.item'  # Movies data
users_path = '../Scientific-Programming/data/ml-100k/u.user'   # Users data

# Load the data using pandas
ratings = pd.read_csv(ratings_path, sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv(movies_path, sep='|', header=None, encoding='latin-1', names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'])
users = pd.read_csv(users_path, sep='|', header=None, encoding='latin-1', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

# Merge the ratings and movie data on 'movie_id'
ratings_movies = pd.merge(ratings, movies[['movie_id', 'movie_title']], on='movie_id')

# Step 1: Visualize the distribution of movie ratings using sns.histplot()
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], kde=True, color='blue', bins=10)
plt.title('Distribution of Ratings in MovieLens 100K Dataset')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Step 2: Bar plot for top genres by average rating
# First, we'll calculate the average rating for each genre
# Create a new DataFrame for the genre data
genre_columns = ['unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']

# Create a new DataFrame for genre-related data
genres = movies[['movie_id'] + genre_columns]

# Merge the ratings data with the genres DataFrame
ratings_genres = pd.merge(ratings_movies, genres, on='movie_id')

# Calculate the average rating for each genre
average_genre_ratings = {}
for genre in genre_columns:
    avg_rating = ratings_genres[ratings_genres[genre] == 1]['rating'].mean()
    average_genre_ratings[genre] = avg_rating

# Create a bar plot for the top genres by average rating
plt.figure(figsize=(12, 6))
sns.barplot(x=list(average_genre_ratings.keys()), y=list(average_genre_ratings.values()), palette='viridis')
plt.title('Top Genres by Average Rating')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()

# Step 3: Heatmap for correlations in movie features
# Convert release_date and video_release_date to datetime and then numeric
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['video_release_date'] = pd.to_datetime(movies['video_release_date'], errors='coerce')

# We can extract year or other numeric values (e.g., time difference) for correlation
movies['release_year'] = movies['release_date'].dt.year
movies['video_release_year'] = movies['video_release_date'].dt.year

# Select relevant columns for the heatmap
corr_data = movies[['release_year', 'video_release_year'] + genre_columns].fillna(0)

# Compute the correlation matrix
corr_matrix = corr_data.corr()

# Plot the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Movie Features')
plt.show()
