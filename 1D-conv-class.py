# done in class Sep 22
# Ian Carr

import numpy as np
import matplotlib.pyplot as plt 

# define spacial domain
nx = 101
dx = 2./(nx-1)
# defin time domain
nt = 25
dt = .02
c = 1. # assumed wave speed

# defining square waves with np.ones
u = np.ones(nx)
u[.5/dx : 1/dx+1]=2 # setting the inial potiion of wave as per IC
print u

# plot the initial square wave
plt.figure()
plt.plot(np.linspace(0,2,nx), u, ls='--')
plt.ylim(0,2.5)

# loops to interate over time then space
un = np.ones(nx) # place holder

for n in range(nt):
	un = u.copy()  # fill the place holder with u values
	for i in range(1,nx):
		u[i] = un[i]-c*dt/dx*(un[i]-un[i-1])

# plotting wave after advancing in time and space
plt.figure()
plt.plot(np.linspace(0,2,nx), u)
plt.ylim(0,2.5)
plt.show()
