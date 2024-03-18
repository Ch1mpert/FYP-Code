from scipy.stats import norm
from math import sqrt, exp
from scipy.stats import norm
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt


def brownian(x0, n, dt, mu, delta, c, out=None):

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    BT = np.zeros(((len(out),len(out[0]))))
    first = []
    local=[]
    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    #np.cumsum(r, axis=-1, out=out)
    #Counter to check if it's reflected or not
    out[0][0]=2*c
    for j in tqdm(range(len(out))):
        max_BT = 0
        #local.append(0)
        first.append(0)
        for i in range(1,len(out[0])):
            BT[j][i]=BT[j][i-1]+np.random.normal(loc=mu*dt, scale = dt**(1/2))
            #if (0 <= (out[j][i]+epsilon) % c <= 2*epsilon): #-> Odd multiples of c
            if max_BT <= BT[j][i]:
                max_BT = BT[j][i]
            out[j][i] = 2*c-(max_BT-BT[j][i]) #math.exp(max_BT - BT[j][i]) - 1
            #if out[j][i]==0:
            #    local[j]+=dt
            #if out[j][i] >= 2*c:
             #   out[j][i]=out[j][i-1]
            # if out[j][i] <= 0:#and first[j] == 0:
            #     #first[j] = dt*i
            #     #break
            #     out[j][i]=out[j][i]
            #     break
    # Add the initial condition.
    #print(sum(local)/len(local))
    #print(first)
    # if sum(first) != 0:
    #     print(sum(first)/sum(x != 0 for x in first))
    #if abs(out) >= 2*c:
      # counter=counter + 1
    #print(between_hitting_time)
    #avg_hitting_time.append(sum(first_hitting_time)/len(first_hitting_time)) #Expected first hitting time
    return out


delta = 1 #delta*sqrt(dt) = standard deviation of Normal distribution. aka sigma/volatility -> Standard brownian has s.d. of sqrt(dt)
# Total time.
T = 10
# Number of steps.
N = 10000
# Time step size
dt = T/N
# Number of realizations to generate.
m = 1
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
#Reflective Bound (Expected first hitting time is 4c^2)
c=2
#Mu is the drift, positive drift of BM reflects into a negative drifted RBM
mu=-0.25
# Initial values of x.
x[:, 0] = 2*c

#Why is it returning a bunch of 1s sometimes?

brownian(x[:,0], N, dt, mu, delta, c, out=x[:,1:])

#print(sum(avg_hitting_time)/len(avg_hitting_time))
t = np.linspace(0.0, N*dt, N+1)
for k in range(m):
    plot(t, x[k], label='2c-RBM with mu = 0.25')
    plot(t, 2*c - x[k], label='RBM with mu = 0.25')
plt.axhline(y = 0,color='red')
plt.axhline(y = 2*c, color='red')
xlabel('Time', fontsize=13)
ylabel('Position', fontsize=13)
plt.legend(loc='upper right')
plt.title("Local time of 2c-RBM at 2c VS Local time of RBM at 0, mu=0.25, c=2, T=10, N=10000")
plt.grid(True)
show()