number = int(input('Enter a number: \n'))
store = []
while number != 1:
    store.append(number)
    if number%2 == 0:
        number = number//2
    else:
        number = (number*3)+1
store.append(1)
print(store)