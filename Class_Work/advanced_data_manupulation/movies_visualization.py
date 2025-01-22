import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../Scientific-Programming/data/movie_dataset.csv")

# Create a seaborn histplot
sns.histplot(data['vote_average'], kde=True, bins=10)

# Set labels and title
plt.xlabel('Movie Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')

# Show the plot
plt.show()


