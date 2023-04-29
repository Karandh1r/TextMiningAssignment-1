def checkprime(n):
    if n <= 1:
        return False
    for i in range(2,n):
        if(n % i == 0):
            return False
    return True    
def getprimeNumbers(n):
    all_primes = []
    for i in range(2,n+1):
        if(checkprime(i)):
            all_primes.append(i)
    return all_primes       
n= 23
print(getprimeNumbers(n))


