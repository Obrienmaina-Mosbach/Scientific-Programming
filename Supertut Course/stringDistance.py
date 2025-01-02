#The edit distance between two strings (s1) and (s2) indicates how many characters in 
# (s1) need to be changed, added, or deleted to transform (s1) into (s2). 
# Given (s1) and (s2) with len(s1) = 3 and len(s2) > 3, 
# calculate the edit distance, considering case sensitivity. 
# (Hint: int(True) = 1, int(False) = 0)

s1 = input("Enter the first string: ")
s2 = "XM5u8IUx"
s2 = s2.lower()
s1 = s1.lower()

char_list1 = list(s1)


if char_list1[0] in s2 and char_list1[1] in s2 and char_list1[2] in s2:
    print(len(s2)-3)

elif char_list1[0] and char_list1[1] in s2:
    print(len(s2)-2)
    
elif char_list1[0] and char_list1[2] in s2:
    print(len(s2)-2)
   
elif char_list1[1] and char_list1[2] in s2:
    print(len(s2)-2)

elif char_list1[0] in s2 or char_list1[1] in s2 or char_list1[2] in s2:
    print(len(s2)-1)
else:
    print(len(s2))

print(char_list1)
