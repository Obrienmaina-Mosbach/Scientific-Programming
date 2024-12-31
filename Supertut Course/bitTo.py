b =[1, 0, 0, 1, 1]

b.reverse()
sum_Total = 0
power = 0

for num in b:
    if power <= len(b):
        result = num*(2**power)
        power += 1
        sum_Total += result
print(sum_Total)