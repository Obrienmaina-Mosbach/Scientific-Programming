liste=[477,778,115,799,385,33,410,800,771,745,510,222,146,107,239,66]

sum_Total = 0

for num in liste:
    for number in range(1, len(liste)):
        sum_Total += (num%number)
print(sum_Total)