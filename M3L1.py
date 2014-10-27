# Convection problems - green light
# Ian Carr

import numpy as np 
import matplotlib.pyplot as plt 

def rho_green_light(nx, rho_max, rho_light):
	"""
	Computes green light IC with shock, linear distribution behind

	Parameters: nx - int - num grid poirts in x
				rho_max - float - max density allowed
				rho_light - float - desity of cars at stoplight

	Returns: rho - array of floats - initial values of desnity
	"""
	rho = np.arange(nx)*2./nx*rho_light # before stoplight
	rho[(nx-1)/2:] = 0

	return rho

# basic inital condition parameters
nx = 81
nt = 30
dx = 4.0/nx

x = np.linspace(0,4,nx)

rho_max = 10.
u_max = 1.
rho_light = 10.

# using desity fn
rho = rho_green_light(nx, rho_max, rho_light)

plt.plot(x, rho)
plt.xlabel('traffic density')
plt.ylabel('distance')
plt.ylim(-0.5,11.)

def computeF(u_max, rho_max, rho):
	"""
	Computes traffic flux F=V*rho

	Parameters: u_max - float - max velocity
				rho - array of float - density
				rho_max - float - max desnity

	Returns: F - array - flux at every point x
	"""
	return u_max*rho*(1-rho/rho_max)

def ftbs(rho,nt,dt,dx,rho_max,u_max):
	"""
	Computes the soln with forward in time, backward in space

	Parameters: rho - array of float - desnity at current time-step
				nt - int - num time steps
				dt - float - time step size
				dx - float - mesh spacing
				rho_max - float - max density
				U - float - speed limit
	Returns: rho_n - array of float - density after timestep n 
	"""
	# initialize our results array with dimenstions nt by nx
	rho_n = np.zeros((nt,len(rho)))
	# copy the inital u array into each row of new array
	rho_n[0,:] = rho.copy()

	for t in range(1,nt):
		F = computeF(u_max, rho_max, rho)
		rho_n[t,1:] = rho[1:] - dt/dx*(F[1:]-F[:-1])
		rho_n[t,0] = rho[0]
		rho_n[t,-1] = rho[-1]
		rho = rho_n[t].copy()

	return rho_n

# stability condition
sigma = 1.

# ------- unstable scheme --------
dt = sigma*dx
rho_n = ftbs(rho,nt,dt,dx,rho_max,u_max)

# -------- up-wind (stable) scheme -------
rho_light = 5.
nt = 40
rho = rho_green_light(nx,rho_max,rho_light)
rho_n = ftbs(rho,nt,dt,dx,rho_max,u_max)

from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0,4),ylim=(-.5,11.5),xlabel=('distance'),\
				ylabel=('traffic density'))
line, = ax.plot([],[])

def animate(data):
	x = np.linspace(0,4,nx)
	y = data
	line.set_data(x,y)
	return line,

plt.show()
