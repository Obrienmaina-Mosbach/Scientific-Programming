a = 678
#sum of digity  of a
numberSplit = [int(digit) for digit in str(a)]
sumTotal = numberSplit[0] + numberSplit[1] + numberSplit[2]
print(str(sumTotal))