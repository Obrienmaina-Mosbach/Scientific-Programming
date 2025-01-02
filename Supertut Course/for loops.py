#iterable object is a collection of items that can be iterated over, such as lists, tuples, and dictionaries.
#iterated over means to loop over the items in the collection.

#for element in iterable_object:
    # Do something with element

numberlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sam_Total = 0
for number in numberlist:
    sam_Total += number
print(sam_Total)

num = []
i = numberlist[0]
j = numberlist[1]
for i in range(i, numberlist[-1], i+1):
    num.append(i + j)
    j += 2
print(num)

mauxed = ['Helloa', 1, 's', 'maize', 2, 't', 'mice', 3, 'u', 'nice', 4, 'v', 'lice', 5, 'w', 'mice', 6, 'x', 'nice', 7, 'y', 'lice', 8, 'z']
onlyString = []
onlyInt = []
for item in mauxed:
    if type(item) == str:
        onlyString.append(item)
    elif type(item) == int:
        onlyInt.append(item)
print(onlyString)
print(onlyInt)

randomtext = 'jskdaQuDmnbcdaduJCSJkfjghdfNamNSDdfdzddsiVCIDbdgxfACS'
collectLower = ''
collectUpper = ''
for letter in randomtext:
    if letter.islower():
        collectLower += letter
    elif letter.isupper():
        collectUpper += letter

print(collectLower)
print(collectUpper)

oceans = ['Atlantic', 'Pacific', 'Indian', 'Southern', 'Arctic']
for indexItem in range(len(oceans)):
    print(f'{indexItem+1}: The {oceans[indexItem]} ocean.')


n = 9
factorial = 1
for i in range(1, n+1):
    factorial *= i
print(factorial)

num = [0, 1]
for sequence in range(2, 9):
    sequence = num[-1] + num[-2]
    num.append(sequence)
print(num)

number = int(input('Enter a number: \n'))
is_Prime = True
for divisor in range(2, number):
    if number % divisor == 0:
        is_Prime = False
        print(f'{number} is a not prime number')
        break
    else:
        print(f'{number} is a prime number')
        break


sum_Total = 0
for number in range(100):
    if number % 2 == 1:
        continue
    sum_Total += number
print(sum_Total)


#Nested Loops
for i in range(3):
    for j in range(4):
        print(f'Element: {i}, {j}')