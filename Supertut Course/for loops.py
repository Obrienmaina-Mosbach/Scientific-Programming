#iterable object is a collection of items that can be iterated over, such as lists, tuples, and dictionaries.
#iterated over means to loop over the items in the collection.

#for element in iterable_object:
    # Do something with element

numberlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sum = 0
for number in numberlist:
    sum += number
print(sum)