age = int(input("Enter your age: \n"))
if age > 10:
    print("You are older than 10 years.")
else:
    print("You are younger than 10 years.")


password = input("Enter your password: \n")
password_Repeat = input("Enter your password again: \n")
if password == password_Repeat:
    print("Passwords Match.")
else:
    print("Passwords do not match.")
    print("Please try again.")

if len(password) < 15:
    password = input("Password must be at least 15 characters long. Please enter a new password: \n")
else:
    print("Password is strong.")
