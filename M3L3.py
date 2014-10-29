# Module 3 Lesson 3 - Traffic flow revisited
# Ian Carr - Oct 29 2014

# this lesson involves lots of symbolic math

import sympy
# assigning symbolic variables
u_max, u_star, rho_max, rho_star, A, B =\
sympy.symbols('u_max u_star rho_max rho_star A B')

# creating symbolic variables
eq1 = sympy.Eq(0,u_max*rho_max*(1-A*rho_max-B*rho_max**2))
eq2 = sympy.Eq(0,u_max*(1-2*A*rho_star-3*B*rho_star**2))
eq3 = sympy.Eq(u_star, u_max*(1-A*rho_star-B*rho_star**2))

# eliminating B in equation 2, creating equation 4
eq4 = sympy.Eq(eq2.lhs - 3*eq3.lhs, eq2.rhs - 3*eq3.rhs)

# getting equations for A from eq1 and eq4
rho_sol = sympy.solve(eq4,rho_star)[0]

B_sol = sympy.solve(eq1,B)[0]

# subbing in rho_sol for rho_star and B_sol for B in eq2
quadA = eq2.subs([(rho_star,rho_sol), (B,B_sol)])
quadAs = quadA.simplify()

# solving for the roots of quadAs
A_sol = sympy.solve(quadAs,A)

# evaluaing soln for A using the values of rho and u from lesson 1
aval = A_sol[0].evalf(subs={u_star:0.7, u_max:1.0, rho_max:10.0})

# doing the same as above for B
bval = B_sol.evalf(subs={rho_max:10.0, A:aval})

# -------- Green light: take 2 --------

# redefining sympy variables as floats to work with numpy
rho_max = 10.
u_max = 1.

def computeF(u_max,rho,aval,bval):
	return u_max*rho*(1-aval*rho-bval*rho**2)

import numpy as np
import matplotlib.pyplot as plt 

def rho_green_light(nx, rho_light):
	# computes greenlight initial condition

	rho_initial = np.arange(nx)*2./nx*rho_light # before stoplight
	rho_initial[(nx-1)/2:] = 0

	return rho_initial

# defining grid size, time steps
nx = 81
nt = 30
dx = 4.0/(nx-1)

x = np.linspace(0,4,nx)
rho_light = 5.5

rho_initial = rho_green_light(nx,rho_light)

plt.plot(x,rho_initial,ls='-',lw=3)
plt.ylim(-0.5,11.)

# defining interative scheme
def ftbs(rho,nt,dt,dx,rho_max,u_max):
	# computes soln with forward time, backward space

	rho_n = np.zeros((nt,len(rho)))
	rho_n[0,:] = rho.copy()

	for t in range(1,nt):
		F = computeF(u_max,rho,aval,bval)
		rho_n[t,1:] = rho[1:] - dt/dx*(F[1:]-F[:-1])
		rho_n[t,0] = rho[0]
		rho_n[t,-1] = rho[-1]
		rho = rho_n[t].copy()

	return rho_n

# CFL
sigma = 1.
dt = sigma*dx

rho_n = ftbs(rho_initial, nt, dt, dx, rho_max, u_max)

# making animation
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0,4),ylim=(-1,8),xlabel=('distance'),\
			ylabel=('traffic density'))
line, = ax.plot([],[],lw=2)

def animate(data):
	x = np.linspace(0,4,nx)
	y = data
	line.set_data(x,y)
	return line,

anim = animation.FuncAnimation(fig,animate,frames=rho_n,interval=50)

plt.show()
