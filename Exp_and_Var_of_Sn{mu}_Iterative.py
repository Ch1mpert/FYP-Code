
import numpy as np
from tqdm import tqdm
import sympy as sp

#Input parameters

mu = -0.5
T = 100
n = 100000
dt = T/n
c=1/(dt**0.5)#math.floor(1/(dt**0.5))
p = (1+(mu)*(dt**(1/2)))/2
q = 1-p
k=n
x=0


#Note that 2*c must be an integer. so this must work for discrete time steps and integers only
results= np.zeros((sp.floor(2*c)+1, n+1))
#pbar = tqdm(total=n)
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
    #pbar.update(0.5)



results1=np.zeros((sp.floor(2*c)+1, n+1)) #how to implement variance like this?

k=n
x=0
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
    

print(results[sp.floor(c),0]*(dt**0.5)) #Exp
print(sp.sqrt(((results1[sp.floor(c),0]*(dt)))-(results[sp.floor(c),0]*(dt**0.5))**2)**2) #s.d.