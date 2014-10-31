# in class exercise Oct 27
# Burgers Equation

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation

def u_initial(nt,nx,dx):
	"""
	makes initial velocity array

	Parameters: nt - int - number of time steps
				nx - int - mesh number
				dx - int - mesh size
	Returns: u0 - array of float - initial values of u
	"""

	u = np.zeros(nx)
	u[0 : 2/(dx+1)] = 1
	print u

	return u

nx = 81
nt = 70
dx = 4.0/nx-1

# computing magnitude of u
computeF = lambda u: (u/2)**2

def maccormack(u,nt,dt,dx,epsilon):
	un = np.zeros((nt,len(u)))
	un[:] = u.copy()
	ustar = u.copy()

	for i in range(1,nt):
		F = computeF(u)
		ustar[1:-2] = u[1:-2] - dt/dx*(F[2:-1]-F[1:-2])\
					+ epsilon*(u[2:-1]-2*u[1:-2]+u[:-3])
		Fstar = computeF(ustar)
		un[i,1:] = .5*(u[1:]+ustar[1:] - dt/dx*(Fstar[1:]-Fstar[:-1]))
		u = un[i].copy()

	return un


# CFL condition
sigma = .5
dt = sigma*dx

# damping coeff
epsilon = .5

def animate(data):
	x = np.linspace(0,4,nx)
	y = data
	line.set_data(x,y)
	return line,


u = u_initial(nt,nx,dt)

un = maccormack(u,nt,dt,dx,epsilon)

fig = plt.figure()
ax = plt.axes(xlim=(0,4), ylim=(-0.5,2))
line, = ax.plot([],[],lw=2)

anim = animation.FuncAnimation(fig,animate,frames=un,interval=50)
plt.show()

