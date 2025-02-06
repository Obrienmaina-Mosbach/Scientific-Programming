x <- 10

if (x > 5) {
    print("x is greater than 5")
} else {
    print("x is 5 or less")
}

for (i in 1:5) {
    print (i)
}

x <- 1
while (x <= 5) {
    print(x)
    x <- x+1
}

square <- function(h){
    return(h^2)
}
print(square(4))