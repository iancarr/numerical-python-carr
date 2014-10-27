# Module 3 Lesson 2 - Numerical schemes for hyperbolic PDEs
# Ian Carr - Oct 20 2014

import numpy as np
import matplotlib.pyplot as plt 

def rho_red_light(nx,rho_max,rho_in):
	"""
	Computes red light IC with shock

	Parameters: nx - int - num grid point
				rho_max - float - max density
				rho_in - float - density of incoming cars

	Returns: rho - array of float - initial values of desnity
	"""

	rho = rho_max*np.ones(nx)
	rho[:(nx-1)*3./4.] = rho_in
	return rho

# IC Parameters
nx = 81
nt = 30
dx = 4.0/nx

rho_in = 5.
rho_max = 10.

V_max = 1. 

x = np.linspace(0,4,nx)

rho = rho_red_light(nx,rho_max,rho_in)

plt.plot(x,rho)
plt.ylabel('traffic density')
plt.xlabel('distance')
plt.ylim(-0.5,11.)
plt.show()