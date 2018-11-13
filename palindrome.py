
def pal(n):
    temp=n
    rev=0
    while(n>0):
        dig=n%10
        rev=rev*10+dig
        n=n//10
    if(temp==rev):
        print("The number is a palindrome!")
        return True
    else:
        print("The number isn't a palindrome!")
        return False



pal(1001)
