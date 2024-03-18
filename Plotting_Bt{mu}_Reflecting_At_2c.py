from scipy.stats import norm
from math import sqrt
from scipy.stats import norm
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean 
import random
import matplotlib.pyplot as plt



def brownian(x0, n, dt, mu, delta, c, out=None, S=None):

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = np.empty(x0.shape + (n,))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.zeros(r.shape)
        S = np.zeros(r.shape)
    p =(1+mu*dt**(1/2))/2
    first = []
    local = []
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    #np.cumsum(r, axis=-1, out=out)
    #Counter to check if it's reflected or not

    for j in range(len(out)):
        #first.append(0)
        #local.append(0)
        hit=0
        for i in range(1,len(out[0])):
            if hit==0:
                test = random.random()
                if test <= p:
                    out[j][i]=out[j][i-1] + dt**(1/2)
                    S[j][i]=S[j][i-1]-dt**(1/2)
                else:
                    out[j][i]=out[j][i-1] - dt**(1/2)
                    S[j][i]=S[j][i-1]+dt**(1/2)
                if out[j][i] <= -2*c or out[j][i] >= 2*c:
                    hit=1
            else:
                out[j][i]=out[j][i-1]
                S[j][i]=S[j][i-1]

    # Add the initial condition.
    #if abs(out) >= 2*c:
      # counter=counter + 1
    #print(between_hitting_time)
    #avg_hitting_time.append(sum(first_hitting_time)/len(first_hitting_time)) #Expected first hitting time
    return out,S


delta = 1 #delta*sqrt(dt) = standard deviation of Normal distribution. aka sigma/volatility -> Standard brownian has s.d. of sqrt(dt)
# Total time.
T = 10
# Number of steps.
N = 1000
# Time step size
dt = T/N
# Number of realizations to generate.
m = 1
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
#Reflective Bound (Expected first hitting time is 4c^2)
c=1
#Mu is the drift of Reflected Walk
mu=-0.25
# Initial values of x.
x[:, 0] = 0

S = np.empty((m,N+1))
S[:, 0] = 0

#Why is it returning a bunch of 1s sometimes?

brownian(x[:,0], N, dt, mu, delta, c, out=x[:,1:], S=S[:,1:])

#print(sum(avg_hitting_time)/len(avg_hitting_time))
t = np.linspace(0.0, N*dt, N+1)
for k in range(m):
    plot(t, x[k]+2*c, label='BM with drift -0.25')
#    plot(t, x[k]+2*c, label='Upper bound of log(Market price)')
#    plot(t, x[k], label='Lower bound of log(Market price)')
    plot(t,S[k]+2*c, label="Reflection about y=2")
xlabel('t')
ylabel('x')
plt.axhline(y = 0,color='red')
plt.axhline(y = 4, color='red')
plt.legend(loc='upper right')
plt.title("BM with drift and its reflection about 2, stopping once hitting either bound of 0 or 4")
grid(True)
show()