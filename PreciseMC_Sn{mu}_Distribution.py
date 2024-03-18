import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean 
import random
import math
from tqdm import tqdm
import sympy as sp


def Sn(N, dt, mu, m, c):
    p=(1+mu*dt**(1/2))/2
    ret=[]
    for j in tqdm(range(m)):
        out=-c
        temp1=0
        temp2=0
        S=0
        for _ in range(N):
            test = random.random()
            if test <= p:
                out=out + 1
            else:
                out=out - 1
            temp1=out
            temp2=out+2*c
            if temp1 >= S:
                S=temp1
            elif temp2<=S:
                S=temp2
        ret.append(S)
        
    return ret

# Total time.
T = 10
# Number of steps.
N = 50
# Time step size
dt = T/N
# Number of realizations to generate.
m = 1000000
#Reflective Bound (Expected first hitting time is 4c^2)
c=math.floor(1/(dt**(1/2)))
#Mu is the drift of Reflected Walk
mu=-0.25
# Initial values of x.


#Find E[S] and Var(S)
p=(1+mu*dt**(1/2))/2
q=1-p
k=N
x=int(c)
results= np.zeros(((2*c)+1, N+1))
while k != 0:
    if k==N:
        for x in range((2*c)+1):
            results[x,k]=0
    for x in range((2*c)+1):  
        if x==0:
            results[x,k-1]=p*results[x+1,k]+q*results[x,k]+q
        elif x==(2*c):
            results[x,k-1]=p*results[x,k]+q*results[x-1,k]-p
        else:
            results[x,k-1]=p*results[x+1,k]+q*results[x-1,k]
    k-=1


k=N
x=int(c)
results1=np.zeros(((2*c)+1, N+1))
while k!=0:
    if k==N:
        results1[x,k]=0
    for x in range((2*c)+1):  
        if x == 0:
            results1[x,k-1] = p * results1[x+1,k] + q * results1[x,k] + q + 2*q*results[0,k]
        elif x==((2*c)):
            results1[x,k-1] = p * results1[x,k] + q * results1[x-1,k] + p - 2*p*results[(2*c),k]
        else:
            results1[x,k-1] = p * results1[x+1,k] + q * results1[x-1,k]
        
    k-=1

mu=0.25
ret=Sn(N, dt, mu, m, c)
#The process starts at 0, and does not get pushed up/down until out hits it, which is the same as R_n starting at [c/sqrt(dt)]

#print(mean(ret))

t = np.linspace(0.0, N*dt, N+1)

U=30
L=-20

probs=[]
for k in range(L,U):
    counted=0
    for i in ret:
        if i==k:
            counted+=1
    probs.append(counted/m)

#print(sum(probs))
#print(results[c,0])
#print(math.sqrt(((results1[c,0])-(results[c,0])**2)))

# plt.hist(ret, bins='auto', density=True)
# plt.title("Path distributions ")

# # Generate x values for the curve
# x_values = np.linspace(L, U, 1000)

# Plot the standard normal distribution curve
# plt.plot(x_values, norm.pdf(x_values, mu*N*dt**0.5, N**(0.5)), label='N(mu*t, t)')

plt.bar(range(L,U), probs)
plt.xlabel('x', fontsize=13)
plt.ylabel('Approx. of P(S_n^mu = x | S_0^mu = [c/sqrt(dt)])', fontsize=13)
plt.title(f'Monte Carlo simulation of the distribution of S_n^mu, with T={T}, {m} paths, dt={T/N}, c=[1/sqrt(dt)], and mu={-mu}')
# Generate x values for the curve
x_values = np.linspace(L, U, 1000)
# Plot the normal distribution curve
plt.plot(x_values, norm.pdf(x_values, float(results[c,0]), float(sp.sqrt(((results1[c,0]))-(results[c,0])**2))), label='N(f(0,[c/sqrt(dt)]), g(0,[c/sqrt(dt)])-f(0,[c/sqrt(dt)])^2)', color='r')
plt.grid(True)
plt.legend()
plt.show()