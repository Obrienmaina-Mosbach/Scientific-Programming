library(dplyr)
file_path <- "../Scientific-Programming/R/data/housing.csv"
df <- read.csv(file_path)
df %>% filter( housing_median_age < 25)
print(head(df))