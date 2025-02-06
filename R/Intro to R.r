sm <- 5+78
print(sm)

prod <- 5*34
print(prod)

div <- 34/6
print(div)

a <- 10
name <- "Jsad"
flag <- TRUE

v <- c(1,2,3,4,5)
u <- c(5,3,7,9,2)

print(v*u)

a <- c(3,4,5,12)
print(median(a))

mat <- matrix(1:14, nrow=2, ncol=7)
print(mat)

# Load a CSV file into a dataframe
file_path <- "../Scientific-Programming/R/data/housing.csv"
df <- read.csv(file_path)

# Display the first few rows of the dataframe
print(head(df))

lst <- list(name="Alo", age=45, scores=c(93, 89, 96))
print(lst)