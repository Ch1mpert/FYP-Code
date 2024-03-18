
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from tqdm import tqdm
import sympy as sp


sys.setrecursionlimit(15000)

mu = -0.5
T = 10
n = 600
dt = T/n
c=0.5/(dt**0.5)
p = (1+(mu)*(dt**(1/2)))/2
q = 1-p


#E[S_n^mu | S_0^mu=c]
k=n
x=0
results= np.zeros((sp.floor(2*c)+1, n+1))
while k != 0:
    if k==n:
        for x in range(sp.floor(2*c)+1):
            results[x,k]=0
    for x in range(sp.floor(2*c)+1):
        if x==0:
            results[x,k-1]=p*results[x+1,k]+q*results[x,k]+q
        elif x==sp.floor(2*c):
            results[x,k-1]=p*results[x,k]+q*results[x-1,k]-p
        else:
            results[x,k-1]=p*results[x+1,k]+q*results[x-1,k]
    k-=1


#E[S_n^mu^2 | S_0^mu=c]
k=n
x=0
results1=np.zeros((sp.floor(2*c)+1, n+1))
while k!=0:
    if k==n:
        results1[x,k]=0
    for x in range(sp.floor(2*c)+1):  
        if x == 0:
            results1[x,k-1] = p * results1[x+1,k] + q * results1[x,k] + q + 2*q*results[0,k]
        elif x==(sp.floor(2*c)):
            results1[x,k-1] = p * results1[x,k] + q * results1[x-1,k] + p - 2*p*results[sp.floor(2*c),k]
        else:
            results1[x,k-1] = p * results1[x+1,k] + q * results1[x-1,k]
        
    k-=1



#P(S_n^mu=x | S_0^mu=x)
memo = {}

def g(n, k, x, c, p, q):
    if n == 0 and k == 0:
        return 1
    if n == 0 and k != 0:
        return 0
    if n < k or n < -k:
        return 0
    if (n, k, x) in memo:
        return memo[n, k, x]
    result = 0
    if x == 0:
        result = p * g(n - 1, k, x + 1, c, p, q) + q * g(n - 1, k - 1, x, c, p, q)
    elif x==math.floor(2*c):
        result = p * g(n - 1, k + 1, x, c, p, q) + q * g(n - 1, k, x - 1, c, p, q)
    else:
        result = p * g(n - 1, k, x + 1, c, p, q) + q * g(n - 1, k, x - 1, c, p, q)
        
    memo[n, k, x] = result
    return result

x=[]
dist=[]
# for i in tqdm(range(math.floor(-7.5/(dt**0.5)), math.floor(20/(dt**0.5)))): #make it such that S_t is between -7.5 and 20
#     x.append(i*(dt**0.5)) #Each time step is dt**0.5, meaning that S_t is i*dt**0.5 -> correct as i is the discrete count S_n
#     y=g(n, i, math.floor(c), math.floor(c), p, q)/(dt**0.5) #why does dividing by dt**0.5 work??????
#     dist.append(y)

for i in tqdm(range(math.floor((-n*math.sqrt(dt)/1.5)), math.floor((1.5*n*sp.sqrt(dt))))): 
    x.append(i)
    y=g(n, i, math.floor(c), c, p, q)
    dist.append(y)

print(sum(dist))
print(error)

# print(sum(dist)*(dt**0.5))

# plt.plot(x, dist)
plt.bar(x, dist)

x_values = np.linspace(-n*math.sqrt(dt)/1.5, 1.5*n*(dt**0.5), 1000)

# x_values = np.linspace(-10, 20, 1000)

plt.plot(x_values, norm.pdf(x_values, float(results[sp.floor(c),0]), float(sp.sqrt(((results1[sp.floor(c),0]))-(results[sp.floor(c),0])**2))), label='N(f(0,[c/sqrt(dt)]), g(0,[c/sqrt(dt)])-f(0,[c/sqrt(dt)])^2)', color='r')

# plt.plot(x_values, norm.pdf(x_values, float(results[sp.floor(c),0]*(dt**0.5)), float(sp.sqrt(((results1[sp.floor(c),0])*dt)-(results[sp.floor(c),0]*(dt**0.5))**2))), label='Normal Distribution', color='r')

# plt.xlabel('x', fontsize=13)
# plt.ylabel('P(S_t=x)', fontsize=13)

plt.xlabel('x', fontsize=13)
plt.ylabel('P(S_n^mu=x | S_0^mu=[c/sqrt(dt)])', fontsize=13)
plt.title(f'Distribution of S_n^mu given S_0^mu=[c/sqrt(dt)], T={T} , n={n}, mu={-mu}')

plt.grid(True)
plt.legend()
plt.show()