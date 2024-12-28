sentence = "books are good for READING and leaRning and \t you can learn something with the 'ß' charcter in langugaes like gErman"
sentence2 = "books"
user_Name = input("Enter your name: ")
print(sentence.capitalize()) # Capitalizes the first letter of the string
print(sentence.casefold()) # Converts the string to lowercase but more aggressive than lower()
print(user_Name.center(30, "_")) # Centers the string in a string of length 30
print(sentence.count("a", 1, 70)) # Counts the number of times a substring "a" appears in a string from index 1 to 70
print(user_Name.startswith("a") and user_Name.endswith("m"))# Checks if the string starts with "a" and ends with "m"
print(sentence.find("ß")) # Finds the index of the first occurrence of the substring "ß"
print(sentence.index("g")) # Finds the index of the first occurrence of the substring "ß"
print(user_Name.isalnum()) # Checks if the string is alphanumeric
print(user_Name.isalpha()) # Checks if the string is alphabetic
print(user_Name.isascii()) # Checks if the string contains ASCII characters
print(sentence.expandtabs(37)) # Expands the tabs in a string to a given number of spaces
print(sentence.replace("and", "&"))
print(len(sentence))
print(sentence[-2]) # Prints the second last character in the string
