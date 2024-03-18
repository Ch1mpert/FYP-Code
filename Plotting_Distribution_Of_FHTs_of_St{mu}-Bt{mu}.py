from scipy.stats import norm
import math
from math import sqrt
from scipy.stats import norm
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean
import random
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

fht = []
local=[]
c=0.5 #with c=1, it should be sth close to 8.7781, and c=2 should be sth around 99.1963, 0.5 would be 1.4365
mu=0.5
T=10#((-1+math.e**(4*c*mu)-4*c*mu)/(2*mu**2))
N=1000000
dt=T/N
k=10000
#var=[]
#counter=0
#a=1

for j in tqdm(range(k)):
    fht.append(0)
    #var.append(0)
    #local.append(0)
    BT = 0
    max_BT = 0
    RBM = 0
    i=0
    #l=0
    while i<N:
        #BT += np.random.normal(loc=mu*dt, scale = dt**(1/2))
        test = random.random()
        if test <= (1+mu*dt**(1/2))/2:
            BT+=dt**(1/2)
        else:
            BT-=dt**(1/2)
        if BT >= max_BT:
            max_BT = BT
        RBM = max_BT - BT
        if RBM >= 2*c:
            fht[j]=(dt*i) #Change to max_BT to check expected increase under expected hitting time.
            break
         #   local[j]+=dt
        #if local[j]>=a and l==0:
         #   counter+=1
          #  l=1
        #if RBM >= c and fht[j]==0:
            #fht[j]=(i*dt)
            #break
            #var[j]=(i*dt-1.43656)**2
        i+=1
   
plt.hist(fht, bins='auto', density=True)
sns.histplot(fht, kde=True, bins='auto')
plt.axvline((sum(fht)/len(fht)), color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text((sum(fht)/len(fht))*1.1, max_ylim*0.9, 'Mean: {:.5f}'.format((sum(fht)/len(fht))))

xlabel('First Hitting Time', fontsize=13)
ylabel('Frequency', fontsize=13)
plt.title("FHTs of 2c of S_n^mu-B_n^mu, mu=0.5, c=0.5, T=10, N=1000000 with 20000 realizations")
plt.show()




#print(sum(fht)/sum(x != 0 for x in fht))

#print(f"The average first hitting time of 1 of the walks simulation RBM is {sum(fht)/sum(x != 0 for x in fht)} under Monte Carlo with {k} walks")

#print(((-1+math.e**(4*c*mu)-4*c*mu)/(2*mu**2)))
#print(1-counter/k)
#print(counter)

#print(sum(var)/(sum(x != 0 for x in var)-1))

#print(sum(x != 0 for x in var)-1)
