#!/bin/python3

N = int(input())
if N % 2 == 0:
    if N >= 2 and N <= 5:
        print ("Not Weird")
    elif N > 5 and N < 21:
        print ("Weird")
    elif N > 20:
        print ("Not Weird")
else:
    print("Weird")
