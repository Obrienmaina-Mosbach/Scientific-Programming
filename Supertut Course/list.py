cars = ['bmw', 'audi', 'toyota', 'subaru']
numbers = [1, 2, 3, 4, 5]
miix = ['bmw', 1, 'audi', 2, 'toyota', 3, 'subaru', 4]
empty = []
alaphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
alaphabet.reverse()
print(alaphabet)

#Remove: Use remove() to remove a specific element from a list.
cars.remove('bmw')

#Pop: Use pop() to remove an element at a specific index from a list.
miix.pop(0)

cars[2] = 'honda'
#Append: Use append() to add an element to the end of a list.
cars.append('ford')

print(miix[-3])
print(cars)


fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
fib.append(fib[-1] + fib[-2])
print(fib)

#Concatenation: Combine lists with the + operator.
print(cars + numbers)
#Repetition: Multiply a list by an integer to repeat it ([1, 'a'] * 3 -> [1, 'a', 1, 'a', 1, 'a']).
print(cars * 3)
#Length: Use len()to get the number of elements in a list.
print(len(fib))
#Sorting: Use sort() to sort the elements of a list.
cars.sort()
#Index: Use index() to find the index of a specific element in a list.
print(cars.index('honda'))
#Membership: Use in to check if an element is in a list.
print(7 in fib)
#Equality: Use == to check if two lists are equal.
print(cars == miix)
#Iteration: Use a for loop to iterate over the elements of a list.

#Slicing: Use a[start:stop:step] to slice a list. Print portions of a list.
print(cars[0:4:2])
num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print(num[1:8:2])# [2, 4, 6, 8] # start at index 1, stop at index 8, step by 2
print(num[2:8:3])# [3, 6] # start at index 2, stop at index 8, step by 3
print(num[2:8:4])# [3, 7] # start at index 2, stop at index 8, step by 4
print(num[2::3])# [3, 6, 9, 12] # start at index 2, stop at the end, step by 3
print(num[:8:3])# [1, 4, 7] # start at the beginning, stop at index 8, step by 3
print(num[::3])# [1, 4, 7, 10] # start at the beginning, stop at the end, step by 3
print(num[::2])# [1, 3, 5, 7, 9, 11] # start at the beginning, stop at the end, step by 2
print(num[::-1])# [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] # start at the end, stop at the beginning, step by -1




#Insert: Use insert() to add an element at a specific index in a list.
cars.insert(0, 'suzuki')


#Clear: Use clear() to remove all elements from a list.
#Copy: Use copy() to create a shallow copy of a list.


#Count: Use count() to count the number of occurrences of an element in a list.
print(cars.count('honda'))
#Extend: Use extend() to add the elements of one list to another list.
cars.extend(alaphabet)
print(cars)
#Max: Use max() to find the maximum element in a list.
print(max(numbers))
#Min: Use min() to find the minimum element in a list.
print(min(alaphabet))
#Sum: Use sum() to find the sum of the elements in a list.
print(sum(numbers))
