# Module 2 Lesson 4 - Burgers' Equation
# Ian Carr Oct 6 2014

import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
from matplotlib import animation
import timeit

# declaring symbolic variables and eqn 
x, nu, t = sp.symbols('x nu t')

# what is phi?
phi = sp.exp(-(x-4*t)**2/(4*nu*(t+1))) + \
sp.exp(-(x-4*t-2*np.pi)**2/(4*nu*(t+1)))

phiprime = phi.diff(x)

# importing magic python fn
from sympy.utilities.lambdify import lambdify

u = -2*nu*(phiprime/phi)+4

ufunc = lambdify((t,x,nu),u)

# print("The value of u at t=1, x=4, nu=3 is {}".format(ufunc(1,4,3)))

# ------- solving burgers eqn ----------
# variable declarations
nx = 101
nt = 100
dx = 2*np.pi/(nx-1)
nu = 0.07 
dt = dx*nu

x = np.linspace(0,2*np.pi,nx)
un = np.empty(nx)
t=0

u = np.asarray([ufunc(t,x0,nu) for x0 in x])

plt.figure(figsize=(8,5),dpi=100)
plt.plot(x,u)
plt.xlim([0,2*np.pi])
plt.ylim([0,10])

# --------- iterating burgers eqn ---------

# comparing computational and analytical soln
for n in range(nt):
	un = u.copy()

	u[1:-1] = un[1:-1] - un[1:-1]*dt/dx*(un[1:-1]-un[:-2]) +\
			+ nu*dt/dx*2*(un[2:]-2*un[1:-1]+un[:-2])

	u[0] = un[0] - un[0] * dt/dx * (un[0]-un[-1]) + nu*dt/dx**2*\
			(un[1] - 2*un[0] + un[-1])
	u[-1] = un[-1] - un[-1] * dt/dx * (un[-1] - un[-2]) + nu*dt/dx**2*\
			(un[0]-2*un[-1] + un[-2])

u_analytical = np.asarray([ufunc(nt*dt,xi,nu) for xi in x])

plt.figure(figsize=(8,5), dpi=100)
plt.plot(x,u, color='k', ls='--', lw=3, label='Computational')
plt.plot(x,u_analytical,label='Analytical')
plt.xlim([0,2*np.pi])
plt.ylim([0,10])
plt.legend()

# --------- animating burger fn ---------

u = np.asarray([ufunc(t,x0,nu) for x0 in x])

fig = plt.figure(figsize=(8,6))
ax = plt.axes(xlim=(0,2*np.pi),ylim=(0,10))
line = ax.plot([],[],color='k',ls='--',lw=3)[0]
line2 = ax.plot([],[],lw=2)[0]
ax.legend(['Computed','Analytical'])

# defining function to solve burgers eqn for animation
def burgers(n):

	un = u.copy()

	u[1:-1] = un[1:-1] - un[1:-1]*dt/dx*(un[1:-1]-un[:-2])+nu*dt/dx**2\
			*(un[2:]-2*un[1:-1]+un[:2])

	u[0] = un[0] - un[0] * dt/dx * (un[0]-un[-1]) + nu*dt/dx**2*\
			(un[2:]-2*un[0]+un[-1])
	u[-1] = un[-1] - un[-1] * dt/dx * (un[-1] - un[-2]) + nu*dt/dx**2*\
			(un[0]-2*un[-1]+un[-2])

	u_analytical = np.asarray([ufunc(n*dt,xi,nu) for xi in x])
	line.set_data(x,u)
	line2.set_data(x,u_analytical)


animation.FuncAnimation(fig,burgers,frames=nt,interval=100)


# ------- testing speed for array operation --------

time = timeit()
u = np.asarray([ufunc(t,x0,nu) for x0 in x])

for n in range(nt):
	un = u.copy()

	for i in range(nx-1):
		u[i] = un[i] - un[i] * dt/dx *(un[i]-un[i-1]) + nu*dt/dx**2*\
				(un[i+1]-2*un[i]+u[i-1])
		u[-1] = un[-1] - un[-1]*dt/dx*(un[-1]-un[-2]) + nu*dt/dx**2*\
				(un[0] - 2*un[-1] + un[-2])



