n = 1
liste=[5,2,4,4,6,6,2,1,3,4,5,6,3,1,2,5,1,4,3,6,2,2,6,5,4,1,2,1,2,2,1,1,1,4,2,3,4,1,3,1,1]
liste.reverse()

#print(liste.count(n)*n)

print(liste.index(n))

print(len(liste))

print(liste.index(n))

b = 4
listi = [5,2,4,4,6,6,2,1,3,4,5,6,3,1,2,5,1,4,3,6,2,2,6,5,4,1,2,1,2,2,1,1,1,4,2,3,4,1,3,1,1]
listi.reverse()
liste = listi[listi.index(b):]
sum_Total = 0
for num in liste:
    sum_Total += num
print(sum_Total)
