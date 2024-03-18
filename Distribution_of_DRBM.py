import sys
from tqdm import tqdm
import math
import sympy as sp
import matplotlib.pyplot as plt

sys.setrecursionlimit(15000)

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


memo = {}

def g(n, k, x, c, p, q):
    if n == 0 and k == x:
        return 1
    if n == 0 and k != x:
        return 0
    if abs(k-x) > n:
        return 0
    if (n, k, x) in memo:
        return memo[n, k, x]
    result = 0
    if x == 0:
        result = p * g(n - 1, k, x + 1, c, p, q) + q * g(n - 1, k, x, c, p, q)
    elif x==2*c:
        result = p * g(n - 1, k, x, c, p, q) + q * g(n - 1, k, x - 1, c, p, q)
    else:
        result = p * g(n - 1, k, x + 1, c, p, q) + q * g(n - 1, k, x - 1, c, p, q)
        
    memo[n, k, x] = result
    return result

#Input parameters

mu = -0.5
T = 10
n = 1000
dt = T/n
c=1/(dt**0.5)
p = (1+(mu)*(dt)**(1/2))/2
q = 1-p
k=0

#P(R_n^mu=k | S_0^mu=c)
print(g(n,k,math.floor(c),math.floor(c),p,q))

#below is for the probability distribution of the endpoints of R_n^(-mu,0,[2c/sqrt(dt)])

x_axis=[]
y = []
for i in tqdm(range(math.floor(2*c)+1)):
    x_axis.append(i)
    y.append(g(n, i, math.floor(c), math.floor(c), p, q))
print(y)


plt.bar(x_axis, y)
plt.xlabel('Endpoints (K)', fontsize=13)
plt.ylabel('Probability', fontsize=13)
addlabels(x_axis, y)
plt.title('Recurrence solution of probability distribution of endpoints of R_n^(-mu,0,6)')
plt.show()

#Below is to calculate the expectation of S_n^mu using only distribution of R_n^mu
Exp=0
for i in tqdm(range(n)):
    Exp+=(g(i, 0, math.floor(c), math.floor(c), p, q)*q-g(i, math.floor(2*c), math.floor(c), math.floor(c), p, q)*p)*(dt**0.5)
print(Exp)

