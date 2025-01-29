# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataset
np.random.seed(42)
data = pd.DataFrame({
    'Square_Footage': np.random.randint(500, 4000, 100),
    'Num_Rooms': np.random.randint(1, 10, 100),
    'Location_Index': np.random.randint(1, 5, 100),
    'Price': np.random.randint(100000, 1000000, 100)
})

print(data)

# Add some correlation
data['Price'] += data['Square_Footage'] * 100  # Price depends strongly on Square Footage
data['Price'] += data['Num_Rooms'] * 5000     # Price depends on Num_Rooms

# Plotting correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Housing Features")
plt.show()


#How to Interpret the Heatmap
#Each cell represents the correlation between two features.
#Strong correlations (close to -1 or 1) are visually distinct:
#Dark red: Strong negative correlation.
#Dark blue: Strong positive correlation.
#Light colors: Weak or no correlation.
#The diagonal will always show a value of 1.0 because each feature is perfectly correlated with itself.

#What is Correlation?
#Correlation measures the linear relationship between two variables. It is represented by a value between -1 and 1:
#1: Perfect positive correlation (as one increases, the other increases).
#-1: Perfect negative correlation (as one increases, the other decreases).
#0: No linear relationship.