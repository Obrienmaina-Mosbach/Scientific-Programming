first_name = "Bildard"
second_name = "Bates"
age = 17

full_Name = first_name + " " + second_name

print("The new student is called " + full_Name + " and he is " + str(age) + " years old.")

#-----lenght of a string-------
print("The length of the full name is: " + str(len(full_Name)))

#-----indexing a string----------
print(first_name[3])

#the last index of a string is length-1 as python is 0 indexed
email_address = input("Enter your email address: \n")
print("We value your privacy and will not share your email address:" + email_address[0]+ "**********" + email_address[-1])