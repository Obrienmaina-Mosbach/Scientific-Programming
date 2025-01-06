prime = []
n = 19

for newNumber in range(2, n):
    is_Prime = True
    for test in range(2, newNumber):
        if newNumber%test == 0:
            is_Prime = False
            
    if is_Prime:
        prime.append(newNumber)
print(prime)


for i in range(2):
    for j in range(2):
        print(f'Element {i}, {j}')
        
