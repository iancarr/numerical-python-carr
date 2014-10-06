# Module 2 Lesson 1, Started Sep 25 2014
# Ian Carr

import numpy as np 
import matplotlib.pyplot as plt

"""
# setting time and space parameters
nx = 80 # number 
dx = 2./(nx-1)
nt = 10 
dt = .02
c = 1 # assuming wave speed of c = 1

# ------- linear convection ---------

# setting wave IC
u = np.ones(nx) # creating space matrix
u[.5/dx : 1/dx+1]=2 # setting u=2 between 0.5 and 1

plt.figure()
plt.plot(np.linspace(0,2,nx), u, ls='--', lw=3)
plt.ylim(0,2.5)


# setting up arrays to interate into
un = np.ones(nx) # initializing array to interate into

for n in range(nt):
	un = u.copy()
	for i in range(1,nx):
		u[i] = un[i] - c*dt/dx*(un[i]-un[i-1])

# plotting interated array

plt.figure()
plt.plot(np.linspace(0,2,nx), u)
plt.ylim(0,2.5)
"""

# -------- non linear convection --------

nx = 81 # number 
dx = 2./(nx-1)
nt = 10 
# dt = .02
c = 1 # assuming wave speed of c = 1
sigma = 0.5

# using the same IC as linear
u = np.ones(nx)
u[.5/dx: 1/dx+1]=2

un = np.ones(nx) # dummy array
x = np.linspace(0,2,nx)

plt.plot(np.linspace(0,2,nx), u)
plt.ylim(0,2.5)

# final time
t_f = 0.0


while t_f<=0.5:
	dt = sigma*dx/(max(u))
	t_f = t_f+dt
	un = u.copy()
	u[1:] = un[1:] - un[1:]*dt/dx*(un[1:]-u[0:-1])
	u[0] = 1.0

plt.plot(np.linspace(0,2,nx), u)
plt.ylim(0,2.5)
plt.show()