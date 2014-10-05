# Module 2 Lesson 3 - 1D Diffusion
# Ian Carr Oct 5 2014

import numpy as np 
import matplotlib.pyplot as plt

# time and space parameters
nx = 41
dx = 2./(nx-1)
nt = 20
nu = 0.3  # kinematic viscosity
sigma = .2
dt = sigma*dx**2/nu

x = np.linspace(0,2,nx)

u = np.ones(nx)
u[.5/dx : 1/dx+1] = 2

# initializing dummy space array
un = np.ones(nx)

# solving central difference for diffusion eqn
for n in range(nt):
	un = u.copy()
	u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:]-2*un[1:-1]\
						+un[0:-2])

# plotting
plt.plot(np.linspace(0,2,nx),u)
plt.ylim(0,2.5)

# --------- Animations --------

# importing animation libraries
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

# initial conditions for animation
nt = 50
u = np.ones(nx)
u[.5/dx : 1/dx+1] = 2

un = np.ones(nx)

# setting up figure and axes for animation
fig = plt.figure(figsize=(8,5))
ax = plt.axes(xlim=(0,2), ylim=(1,2.5))
line = ax.plot([],[], ls='--', lw=3)[0]

# defining func to solve for plots at each time
def diffusion(i):
	line.set_data(x,u)

	un=u.copy()
	u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:]-2*un[1:-1]+\
				un[0:-2])

animation.FuncAnimation(fig,diffusion,frames=nt,interval=100)

plt.show()