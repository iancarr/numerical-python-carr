# Module 2 Lesson 2: CFL Condition
# Ian Carr Sep 29 2014

import numpy as np 
import matplotlib.pyplot as plt 

# linear convection function
def linearconv(nx):
	"""
	Solves linear convection eqn
	d_t u + c d_x = 0
	- the wave speed is set at 1
	- domain in x: [0,2]
	- 20 timesteps with dt = 0.025
	- IC: hat function

	plots results

	Parameter: nx - int - number of internal grid points
	Returns: none
	"""
	# IC
	dx = 2./(nx-1)
	nt = 20
	dt = 0.025
	c = 1

	# Hat fn
	u = np.ones(nx)
	u[.5/dx : 1/dx+1] = 2

	un = np.ones(nx)

	for n in range(nt):
		un = u.copy()
		u[1:] = un[1:] - c*dt/dx*(un[1:] - un[0:-1])
		u[0] = 1.0

	plt.figure
	plt.plot(np.linspace(0,2,nx), u)
	plt.ylim(0,2.5)
	plt.show()

# using linear convection fn
# linearconv(41)

# ---------- code rewritten with the CFL condition --------

def linearconv(nx):
	"""
	solve linear convection eqn

	same as fn above with CFL condition
	- dt computed using CFL of 0.5

	Parameters: nx - int - internal grid points
	Returns: none
	"""

	# IC
	dx = 2./(nx-1)
	nt = 20
	c = 1
	sigma = .5 # CFL condition

	dt = sigma*dx

	# hat fn
	u = np.ones(nx)
	u[.5/dx : 1/dx+1]=2

	un = np.ones(nx)

	for n in range(nt):
		un = u.copy()
		u[1:] = un[1:] - c*dt/dx*(un[1:] - un[0:-1])
		u[0] = 1.0

	plt.figure()
	plt.plot(np.linspace(0,2,nx), u)
	plt.ylim(0,2.5)
	plt.show()

# testing modified linear convection fn
linearconv(41)
linearconv(75)
linearconv(1000)
