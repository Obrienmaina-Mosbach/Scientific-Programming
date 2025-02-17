library(ggplot2)
library(dplyr)

file_path <- "../Scientific-Programming/data/movie_dataset.csv"
df <- read.csv(file_path)
print(head(df))

# Check the structure of the dataframe to understand the available columns
print(str(df))

# Write the dataframe to a CSV file
write.csv(df, "output.csv", row.names = FALSE)

# Plot budget vs revenue
plot <- ggplot(df, aes(x=budget, y=revenue)) + 
  geom_point() +
  labs(title="Budget vs Revenue",
       x="Budget",
       y="Revenue") +
  theme_minimal()

print(plot)

# Apply summary to mtcars dataset and print the summary
print(summary(mtcars))


# Create a linear model
model <- lm(mpg ~ hp, data=mtcars)

# Print the summary of the linear model
print(summary(model))