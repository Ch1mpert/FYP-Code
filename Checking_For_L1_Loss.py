
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from tqdm import tqdm
import sympy as sp
import scipy.stats
import seaborn as sns


sys.setrecursionlimit(15000)



#Estimated Absolute Error

range_n=[]
x_axis=[]
sum_diff=[]
sumg=[]


for j in tqdm(range(1,51)):
    x_axis.append(j)
    mu = -0.25
    T = j
    n = j**2
    dt = T/n
    c = 1/(dt**0.5)
    p = (1+(mu)*(dt**(1/2)))/2
    q = 1-p

    #P(S_n^mu=x | S_0^mu=x)
    memo={}
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
        elif x==2*c:
            result = p * g(n - 1, k + 1, x, c, p, q) + q * g(n - 1, k, x - 1, c, p, q)
        else:
            result = p * g(n - 1, k, x + 1, c, p, q) + q * g(n - 1, k, x - 1, c, p, q)
            
        memo[n, k, x] = result
        return result

    #E[S_n^mu | S_0^mu=c]
    k=n
    x=0
    results=np.zeros((sp.floor(2*c)+1, n+1))
    while k != 0:
        for x in range(sp.floor(2*c)+1):
            if x==0:
                results[x,k-1]=p*results[x+1,k]+q*results[x,k]+q
            elif x==(2*c):
                results[x,k-1]=p*results[x,k]+q*results[x-1,k]-p
            else:
                results[x,k-1]=p*results[x+1,k]+q*results[x-1,k]
        k-=1


    #E[S_n^mu^2 | S_0^mu=c]
    k=n
    x=0
    results1=np.zeros((sp.floor(2*c)+1, n+1))
    while k!=0:
        for x in range(sp.floor(2*c)+1):  
            if x == 0:
                results1[x,k-1] = p * results1[x+1,k] + q * results1[x,k] + q + 2*q*results[0,k]
            elif x==(2*c):
                results1[x,k-1] = p * results1[x,k] + q * results1[x-1,k] + p - 2*p*results[sp.floor(2*c),k]
            else:
                results1[x,k-1] = p * results1[x+1,k] + q * results1[x-1,k]
            
        k-=1
    diff=0
    count=0
    sumprob=0
    for i in range(math.floor((-n*math.sqrt(dt))*1.5), math.floor((1.5*n*sp.sqrt(dt)))):
        #count+=scipy.stats.norm(float(results[sp.floor(c),0]), float(sp.sqrt(((results1[sp.floor(c),0]))-(results[sp.floor(c),0])**2))).pdf(i)
        #sumprob += g(n, i, math.floor(c), math.floor(c), p, q)
        diff+=(abs(scipy.stats.norm(float(results[sp.floor(c),0]), float(sp.sqrt(((results1[sp.floor(c),0]))-(results[sp.floor(c),0])**2))).pdf(i)-g(n, i, math.floor(c), math.floor(c), p, q)))
    #range_n.append(count)
    #sumg.append(sumprob)
    sum_diff.append(diff)

#print(range_n)
#print(sumg)
print(sum_diff)

plt.bar(x_axis, sum_diff)
sns.histplot(x=x_axis, weights=sum_diff, discrete=True, kde=True, bins='auto')
plt.xlabel('sqrt(Number of steps)', fontsize=13)
plt.ylabel('Estimated L1-loss', fontsize=13)
#plt.axhline(y = 0.2, color='red')
plt.title("Estimated L1-loss with increasing t and n, n=t^2, mu=0.25")
#plt.title("Estimated L1-loss with t=10 and n<=650")
plt.show()