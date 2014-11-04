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

u_max = 1. 

x = np.linspace(0,4,nx)

rho = rho_red_light(nx,rho_max,rho_in)

print rho

plt.plot(x,rho)
plt.ylabel('traffic density')
plt.xlabel('distance')
plt.ylim(-0.5,11.)

# ----- functions for animation -------
def computeF(u_max, rho_max, rho):
	"""
	computes flux f = V*rho

	Parameters: u_max - float - max velocity allowed
				rho_max - float - max density allowed
				rho - array of float - density at every point x

	Returns: F - array - flux at every point
	"""
	return u_max*rho*(1-rho/rho_max)

from matplotlib import animation

def animate(data):
	x = np.linspace(0,4,nx)
	y = data
	line.set_data(x,y)
	return line,


# ----- lax-friedrichs scheme -------
def laxfriedrichs(rho,nt,dt,dx,rho_max,u_max):
	"""
	computes the soln with lax-friedrichs

	Parameters: rho - array of float - current density
				nt - int - number of time steps
				dt - float - time-step size
				dx - float - mesh spacing
				rho_max - float - max allowed car density
				u_max - float - speed limit

	Return: rho_n - array of float - density at all points
	"""
	# initialize results array with nt and nx
	rho_n = np.zeros((nt,len(rho)))
	# copy the initial u array into each row of array
	rho_n[:,:] = rho.copy()

	for t in range(1,nt):
		F = computeF(u_max, rho_max, rho)
		rho_n[t,1:-1] = .5*(rho[2:]+rho[:-2])-\
						dt/(2*dx)*(F[2:]-F[:-2])
		rho_n[t,0] = rho[0]
		rho_n[t,-1] = rho[-1]
		rho = rho_n[t].copy()

	return rho_n

# test with CFL = 1
"""
sigma = 1.0
dt = sigma*dx

rho = rho_red_light(nx, rho_max, rho_in)
rho_n = laxfriedrichs(rho,nt,dt,dx,rho_max,u_max)
"""
# test with CFL = 0.5
sigma = 0.5
dt = sigma*dx

rho = rho_red_light(nx,rho_max,rho_in)
rho_n = laxfriedrichs(rho,nt,dt,dx,rho_max,u_max)
"""
# plotting
fig = plt.figure()
ax = plt.axes(xlim=(0,4),ylim=(4.5,11),xlabel=('distance'),\
				ylabel=('traffic density'))
line, = ax.plot([],[],lw=2)
anim = animation.FuncAnimation(fig,animate,frames=rho_n,\
	interval=50)
"""
# -------- Lax Wendroff scheme -------
def Jacobian(u_max, rho_max, rho):
	return u_max*(1-2*rho/rho_max)

def laxwendroff(rho,nt,dt,dx,rho_max,u_max):
	# initial results array
	rho_n = np.zeros((nt,len(rho)))
	rho_n[:,:] = rho.copy()

	for t in range(1,nt):
		F = computeF(u_max, rho_max, rho)
		J = Jacobian(u_max, rho_max, rho)
		rho_n[t,1:-1] = rho[1:-1] - dt/(2*dx)*(F[2:]-F[:-2])- \
        dt**2/(4*dx**2) * ((J[2:]+J[1:-1])*\
        (F[2:]-F[1:-1]) - (J[1:-1]+J[:-2])*\
        (F[1:-1]-F[:-2]))
        rho_n[t,0] = rho[0]
        rho_n[t,-1] = rho[-1]
        rho = rho_n[t].copy()

	
	return rho_n

"""
# Lax-Whendroff with CFL = 1
rho = rho_red_light(nx, rho_max, rho_in)
sigma = 1
dt = sigma*dx
rho_n = laxwendroff(rho,nt,dt,dx,rho_max,u_max)
"""
# Lax-Whendroff with CFL = 0.5
rho = rho_red_light(nx, rho_max, rho_in)
sigma = 0.5
dt = sigma*dx
rho_n = laxwendroff(rho,nt,dt,dx,rho_max,u_max)
"""
fig = plt.figure()
ax = plt.axes(xlim=(0,4),ylim=(4.5,11),xlabel=('distance'),\
				ylabel=('traffic density'))
line, = ax.plot([],[],lw=2)
anim = animation.FuncAnimation(fig,animate,frames=rho_n,\
	interval=50)
"""

# -------- MacCormack Scheme ---------


def maccormack(rho,nt,dt,dx,u_max,rho_max):
	rho_n = np.zeros((nt,len(rho)))
	rho_star = np.empty_like(rho)
	rho_n[:,:] = rho.copy()
	rho_star = rho.copy()

	for t in range(1,nt):
		F = computeF(u_max,rho_max,rho)
		rho_star[:-1] = rho[:-1] - dt/dx * (F[1:]-F[:-1])
		Fstar = computeF(u_max,rho_max,rho_star)
		rho_n[t,1:] = .5*(rho[1:]+rho_star[1:]-dt/dx*\
			(Fstar[1:] - Fstar[:-1]))
		rho = rho_n[t].copy()
	return rho_n

rho = rho_red_light(nx, rho_max, rho_in)
sigma = 0.5
dt = sigma*dx

rho_n = maccormack(rho,nt,dt,dx,u_max,rho_max)

fig = plt.figure();
ax = plt.axes(xlim=(0,4),ylim=(4.5,11),xlabel=('distance'),\
				ylabel=('traffic density'));
line, = ax.plot([],[],lw=2);

anim = animation.FuncAnimation(fig, animate, frames=rho_n,\
								interval=50)
plt.show()


