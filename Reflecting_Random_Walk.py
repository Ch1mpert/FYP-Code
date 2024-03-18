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


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


def random_walk(steps, p, dt, num_walks, x, T):
    all_positions = []
    end=[]
    count = []
    fht=[]
    for i in tqdm(range(num_walks)):
        position = math.floor(x)
        positions = [position]
        count.append(0)
        for _ in range(steps):
            test=random.random()
            if position == 0:
                if test <= p:
                    position += 1
                else:
                    position = position
                    count[i]+=1
                
            elif position == math.floor(2*x):
                if test <= p:
                    position = position
                    count[i]-=1
                else:
                    position -= 1
            else:
                if test <= p:
                    position += 1
                else:
                    position -= 1

            positions.append(position)
        end.append(positions[-1])
        
    return all_positions, fht, end, count

#Input parameters

steps = 100
mu = -0.5
T = 10
dt = T / steps
p = (1 + mu * (dt ** 0.5)) / 2
num_walks = 10000
c = math.floor(1/(dt**0.5))

all_positions, fht, end, Exp= random_walk(steps, p, dt, num_walks, math.floor(c), T)

#The following estimates P(R_n=k | R_0=c)

x_axis=[]
probs=[]
counted=np.zeros(math.floor(2*c)+1)
for k in range(0, math.floor(2*c)+1):
    x_axis.append(k)
    for i in end:
        if i==k:
            counted[k]+=1
    probs.append(counted[k]/num_walks)
        
print(probs)
plt.bar(x_axis, probs)
plt.xlabel('Endpoints (K)', fontsize=13)
plt.ylabel('Probability', fontsize=13)
addlabels(x_axis, probs)
plt.title('Monte Carlo of probability distribution of endpoints of R_n^(-mu,0,6)')
plt.show()
