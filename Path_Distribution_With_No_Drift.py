import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
from scipy.stats import norm
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean 
import random
from tqdm import tqdm
import numpy as np
from scipy.stats import norm



def brownian(x0, n, dt, mu, delta, c, out=None, S=None):

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
        S = np.empty(r.shape)
    p =(1+mu*dt**(1/2))/2
    first = []
    local = []
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    #np.cumsum(r, axis=-1, out=out)
    #Counter to check if it's reflected or not

    for j in tqdm(range(len(out))):
        #first.append(0)
        #local.append(0)
        temp1=0
        temp2=0
        for i in range(1,len(out[0])):
            test = random.random()
            if test <= p:
                out[j][i]=out[j][i-1] + dt**(1/2)
                
            else:
                out[j][i]=out[j][i-1] - dt**(1/2)
            temp1=out[j][i]-c
            temp2=out[j][i]+c
            if temp1 > S[j][i-1]:
                S[j][i]=temp1
            elif temp2<S[j][i-1]:
                S[j][i]=temp2
            else:
                S[j][i]=S[j][i-1]

    # Add the initial condition.
    #if abs(out) >= 2*c:
      # counter=counter + 1
    #print(between_hitting_time)
    #avg_hitting_time.append(sum(first_hitting_time)/len(first_hitting_time)) #Expected first hitting time
    return out,S


delta = 1 #delta*sqrt(dt) = standard deviation of Normal distribution. aka sigma/volatility -> Standard brownian has s.d. of sqrt(dt)
# Total time.
T = 100
# Number of steps.
N = 10000
# Time step size
dt = T/N
# Number of realizations to generate.
m = 10000
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
#Reflective Bound (Expected first hitting time is 4c^2)
c=1
#Mu is the drift of Reflected Walk
mu=0
# Initial values of x.
x[:, 0] = 0

S = np.empty((m,N+1))
S[:, 0] = 0

#Why is it returning a bunch of 1s sometimes?

brownian(x[:,0], N, dt, mu, delta, c, out=x[:,1:], S=S[:,1:])

x=[]
for k in range(m):
    x.append((S[k][-1]))#(c*(8*T)**(0.5)))
print(len(x))
plt.hist(x, bins='auto' ,density=True)
plt.title("Path distributions of S_T, with T=100, 10000 paths, c=1, and dt=0.01")

# Generate x values for the curve
x_values = np.linspace(-40, 40, 1000)

# Plot the standard normal distribution curve
plt.plot(x_values, norm.pdf(x_values, 0, T**0.5), label='N(0, t)')


plt.legend()
plt.show()

#print(sum(avg_hitting_time)/len(avg_hitting_time))
t = np.linspace(0.0, N*dt, N+1)
#for k in range(m):
    #plot(t, x[k]+c, label='log(Market price)')
    #plot(t, x[k]+2*c, label='Upper bound of log(Market price)')
    #plot(t, x[k], label='Lower bound of log(Market price)')
    #plot(t,S[k]/(c*(8*T)**(0.5)))#, label="log(LP price)")
#xlabel('t')
#ylabel('x')
#plt.title("S_T/(c*(8T)^0.5)")
#grid(True)
#show()