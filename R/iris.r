library(ggplot2)
library(dplyr)
library(caret)

# Load Iris dataset
data(iris)
print(head(iris))

# Check the structure of the Iris dataset to understand the available columns
print(str(iris))

# Write the Iris dataset to a CSV file
write.csv(iris, "iris_output.csv", row.names = FALSE)

# Plot Sepal.Length vs Sepal.Width
plot_iris <- ggplot(iris, aes(x=Sepal.Length, y=Sepal.Width, color=Species)) + 
  geom_point() +
  labs(title="Sepal Length vs Sepal Width",
       x="Sepal Length",
       y="Sepal Width") +
  theme_minimal()

print(plot_iris)

# Apply summary to the Iris dataset and print the summary
print(summary(iris))

# Create a linear model using caret
model <- train(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data=iris, method="lm")

# Print the summary of the linear model
print(summary(model$finalModel))