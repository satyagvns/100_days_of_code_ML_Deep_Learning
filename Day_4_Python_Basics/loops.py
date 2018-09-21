'''
Read an integer . For all non-negative integers , print . See the sample for details.

Input Format

The first and only line contains the integer, .

Constraints


Output Format

Print  lines, one corresponding to each .
'''

if __name__ == '__main__':
    n = int(input())
    if n > 0:
        for x in range(0, n):
            print(x**2)
    else:
        print("Operation Not Possible as per condition")
