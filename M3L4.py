# Module 3 Lesson 4 - Finite volume method
# Ian Carr Nov 15

from traffic import rho_red_light, computeF

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation


# initial conditions
nx = 101
nt = 30
dx = 4.0/(nx-2)

rho_in = 5.
rho_max = 10.

V_max = 1.

x = np.linspace(0,4,nx-1)

rho = rho_red_light(nx-1, rho_max, rho_in)

def animate(data):
	x = np.linspace(0,4,nx-1)
	y = data
	line.set_data(x,y)
	return line,

# ---------- Godunov's method ----------

def godunov(rho, nt, dt,dx,rho_max,V_max):
	"""
	Computes the soln with Godunov scheme using 
	Lax-Friedrichs flux.
	"""

	# initialize results array
	rho_n = np.zeros((nt,len(rho)))
	# copy initial array into each row of array
	rho_n[:,:] = rho.copy()

	# setup temporary arrays
	rho_plus = np.zeros_like(rho)
	rho_minus = np.zeros_like(rho)
	flux = np.zeros_like(rho)

	for t in range(1,nt):

		rho_plus[:-1] = rho[1:] # cell boundary instead
		rho_minus = rho.copy()
		flux = 0.5 * (computeF(V_max, rho_max, rho_minus)+\
					computeF(V_max,rho_max,rho_plus)+\
					dx/dt * (rho_minus-rho_plus))
		rho_n[t,1:-1] = rho[1:-1] + dt/dx*(flux[:-2]-\
						flux[1:-1])
		rho_n[t,0] = rho[0]
		rho_n[t,-1] = rho[-1]
		rho = rho_n[t].copy()

	return rho_n

# CFL condition
sigma = 1.0
dt = sigma*dx/V_max

rho = rho_red_light(nx-1,rho_max, rho_in)
rho_n = godunov(rho,nt,dt,dx,rho_max,V_max)

# plotting
fig = plt.figure()
ax = plt.axes(xlim=(0,4), ylim=(4.5,11),xlabel=('distance'),\
			ylabel=('traffic density'))
line, = ax.plot([],[],lw=2)

anim = animation.FuncAnimation(fig,animate,frames=rho_n,\
								interval=50)




# --------- MUSCL Scheme --------



def minmod(e,dx):
	"""
	Computes the minmod approximation of the slope
	"""

	sigma = np.zeros_like(e)
	de_plus = np.ones_like(e)
	de_minus = np.ones_like(e)

	de_minus[1:] = (e[1:] - e[:-1])/dx
	de_plus[:-1] = (e[1:] - e[:-1])/dx
	
	# the following is inefficient but easy to read
	for i in range(1,len(e)-1):
		if (de_minus[i]*de_plus[i]<0.0):
			sigma[i] = 0.0
		elif (np.abs(de_minus[i]) < np.abs(de_plus[i])): 
			sigma[i] = de_minus[i]
		else:
			sigma[i] = de_plus[i]

	return sigma

def muscl(rho,nt,dt,dx,rho_max,V_max):
	"""
	Computes the soln with MUSCL scheme using the\
	Lax-Friedrichs flux and RK2 method with limited slope
	"""
	# iniitalize results array
	rho_n = np.zeros((nt,len(rho)))
	# copy the initial array into each row of array
	rho_n[:,:] = rho.copy()

	# temporary arrays
	rho_plus = np.zeros_like(rho)
	rho_minus = np.zeros_like(rho)
	flux = np.zeros_like(rho)
	rho_star = np.zeros_like(rho)

	for t in range(1,nt):

		sigma = minmod(rho,dx) # minmod slope

		# values at cell boundaries
		rho_left = rho + sigma*dx/2.
		rho_right = rho - sigma*dx/2.

		flux_left = computeF(V_max,rho_max,rho_left)
		flux_right = computeF(V_max,rho_max,rho_right)

		# flux i = i + 1/2
		flux[:-1] = 0.5 * (flux_right[1:] + flux_left[:-1] -\
			dx/dt * (rho_right[1:] - rho_left[:-1]))

		# RK2 step 1
		rho_star[1:-1] = rho[1:-1] + dt/dx * (flux[:-2] - \
			flux[1:-1])

		rho_star[0] = rho[0]
		rho_star[-1] = rho[-1]

		sigma = minmod(rho_star,dx)

		# reconstruct values at cell boundary
		rho_left = rho_star + sigma *dt/2.
		rho_right = rho_star - sigma*dt/2.

		flux_left = computeF(V_max, rho_max, rho_left)
		flux_right = computeF(V_max, rho_max, rho_right)

		flux[:-1] = 0.5 * (flux_right[1:] + flux_left[:-1] -\
				dx/dt*(rho_right[1:] - rho_left[:-1]))

		rho_n[t,1:-1] = 0.5 * (rho[1:-1] + rho_star[1:-1] + \
			dx/dt * (flux[:-2] - flux[1:-1]))

		rho_n[t,0] = rho[0]
		rho_n[t,-1] = rho[-1]
		rho = rho_n[t].copy()

		return rho_n

sigma = 1.
dt = sigma*dx/V_max
rho = rho_red_light(nx-1,rho_max,rho_in)
rho_n = muscl(rho,nt,dt,dx,rho_max,V_max)

fig = plt.figure()
ax = plt.axes(xlim=(0,4),ylim=(4.5,11),xlabel=('distance'),\
	ylabel=('traffic density'))
line, = ax.plot([],[],lw=2)

anim = animation.FuncAnimation(fig,animate,frames=rho_n,\
								interval=50)

plt.show()