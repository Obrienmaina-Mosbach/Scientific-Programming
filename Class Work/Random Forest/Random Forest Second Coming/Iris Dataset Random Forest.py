import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
# Iris dataset is a preloaded dataset in scikit-learn, widely used for classification tasks
iris = load_iris()

# Convert to a pandas DataFrame for easier manipulation
data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
data['target'] = iris['target']  # Adding the target column with labels

# Mapping target integers to class names for readability
data['target_name'] = data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Exploratory Data Analysis (EDA)
# Visualizing the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Visualizing the distribution of the target variable
sns.countplot(x='target_name', data=data)
plt.title("Distribution of Target Classes")
plt.show()

# Split the data into features (X) and target (y)
X = data[iris['feature_names']]  # Features
y = data['target']            # Target variable

# Splitting the data into training and testing sets
# Training set: 80%, Testing set: 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Random Forest Classifier**
# Random Forest is an ensemble learning method that builds multiple decision trees
# and combines their predictions to improve accuracy and reduce overfitting.

# Initialize the Random Forest Classifier
# n_estimators: Number of decision trees in the forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Evaluate the model
# Confusion Matrix: Shows how well the model predicts each class
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report: Provides precision, recall, and F1-score for each class
# Precision, Recall, and F1-Score are metrics used to evaluate classification models.
# They help us understand how well the model is performing in predicting the correct labels.

# 1. **Precision**:
#    Precision focuses on the "positive" predictions made by the model.
#    It answers: "Of all the instances the model predicted as positive, how many were actually correct?"
#    Formula: Precision = True Positives / (True Positives + False Positives)
#    Example: If the model says 10 people have a disease, but only 7 actually do,
#    precision = 7/10 = 0.7 or 70%.

# 2. **Recall (or Sensitivity)**:
#    Recall focuses on finding all the positive cases.
#    It answers: "Of all the actual positive cases, how many did the model correctly identify?"
#    Formula: Recall = True Positives / (True Positives + False Negatives)
#    Example: If 10 people actually have a disease and the model identifies 8 of them correctly,
#    recall = 8/10 = 0.8 or 80%.

# 3. **F1-Score**:
#    F1-Score combines Precision and Recall into a single metric using their harmonic mean.
#    It balances the trade-off between Precision and Recall.
#    Formula: F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
#    Example: If Precision = 0.7 and Recall = 0.8, F1-Score = 2 * (0.7 * 0.8) / (0.7 + 0.8) = 0.746 or ~75%.
#    F1-Score is especially useful when you need a balance between Precision and Recall.

# Why are these metrics important?
# - If Precision is high but Recall is low, it means the model is cautious and predicts only when it's confident,
#   but might miss many actual positives (e.g., skipping people who truly have a disease).
# - If Recall is high but Precision is low, it means the model predicts a lot of positives,
#   but many of these are false positives (e.g., incorrectly diagnosing healthy people).
# - F1-Score helps to measure the overall performance when both Precision and Recall matter.

# Printing the classification report shows these metrics for each class in the dataset.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris['target_names']))

# Feature Importance: Identifies which features are most important in making predictions
# Random Forest provides a built-in feature to calculate feature importance
importance = rf.feature_importances_

# Display feature importance
print("\nFeature Importance:")
for feature, importance_score in zip(iris.feature_names, importance):
    print(f"{feature}: {importance_score:.4f}")


# Visualize feature importance
sns.barplot(x=importance, y=iris.feature_names)
plt.title("Feature Importance in Random Forest")
plt.show()

