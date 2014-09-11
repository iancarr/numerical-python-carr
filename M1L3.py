# Module 1 Lesson 3 - Full Phugoid Model
# Ian Carr Sep 11 2014

from math import sin, cos, log, ceil
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# model parameters
g = 9.81 	# gravity (m/s)
v_t = 30.0 	# trim velocity (m/s)
C_D = 1/40. # drag coeff
C_L = 1.0  	# lift coeff - 1 for conveinence

# ------- set initial conditions -------
v0 = v_t		# start at trim velocity
theta0 = 0.0 	# inital angle of trajectory
x0 = 0.0 		# horisontal position (arbitrary)
y0 = 1000.0		# initial altitude

# --------- system of equations --------

# define function for sys of eqn
def f(u):
	"""returns the right-hand side of the phugoid sys of eqn

	Parameters - u: array of floats
					array containing the solution at n

	Returns - dudt: array of float
					array containing the RHS for u
	"""

	v = u[0]
	theta = u[1]
	x = u[2]
	y = u[3]

	return np.array([-g*sin(theta) - C_D/C_L*g/v_t**2*v**2,\
					-g*cos(theta)/v + g/v_t**2*v,\
					v*cos(theta),\
					v*sin(theta)])


# solve system using eulers method

def euler_step(u,f,dt):
	"""returns the solution at the next time-step

	Parameters - u: array of float
					solution to the prev time-step
				f: function
					funciton to compute the RHS of the sys
				dt: float
					time-step

	Returns: u_n_plus_1 : array of float
						approx soln at the next time-step
    """

	return u + dt*f(u) 
    
# ------- solve the system to get trajectory -------
 
T = 100.0			# final time
dt = 0.1 			# time-step
N = int(T/dt) + 1 	# number of time-steps
t = np.linspace(0.0,T,N) # discretized time

# initialize the array containing soln for each time-step
u = np.empty((N,4))
u[0] = np.array([v0,theta0,x0,y0]) # fill the 1st element 

# time loop - Euler Method
for n in range(N-1):
	u[n+1] = euler_step(u[n],f,dt)

# -------- plot the trajectory ------

# get position from soln array
x = u[:,2]
y = u[:,3]

# plot the position information
plt.figure(figsize=(8,6))
plt.grid(True)
plt.xlabel(r'x',fontsize=18)
plt.ylabel(r'y',fontsize=18)
plt.title('Glider trajectory, flight time = %.2f' % T,\
	fontsize=18)
plt.plot(x,y,'k-',lw=2)


# --------- test convergence --------
# testing convergence by varying the value of dt

dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])

u_values = np.empty_like(dt_values, dtype=np.ndarray)

for i, dt in enumerate(dt_values):
	N = int(T/dt)+1 	# number of time steps

	## discretize time t ##
	t = np.linspace(0.0,T,N)

	# initialize the array containing the soln at each time
	u = np.empty((N,4))
	u[0] = np.array([v0, theta0, x0, y0])

	# time loop
	for n in range(N-1):
		u[n+1] = euler_step(u[n],f,dt) # call euler_step fn 

	# store soln values in grid
	u_values[i] = u

# make a functions to compare trajectories with diff dt
def get_diffgrid(u_current, u_fine, dt):
	""" 
	returns the difference between grid and finest grid with
	L_1 norm

	Parameters - u_current: array of float
							soln on current grid
				u_finest: array of float
							soln on the fine grid
				dt: float, time-step

	Returns - diffgrid: float
					computed difference between grids
	"""

	N_current = len(u_current[:,0])
	N_fine = len(u_fine[:,0])

	grid_size_ratio = ceil(N_fine/float(N_current))

	diffgrid = dt *np.sum(np.abs(\
		u_current[:,2]- u_fine[::grid_size_ratio,2]))
	return diffgrid

# computing the differnce between grid and fine grid
diffgrid = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):
	print('dt={}'.format(dt))

	# call the function to compute grid difference
	diffgrid[i] = get_diffgrid(u_values[i], u_values[-1], dt)

# ------- plot the grid convergence results ------

# log-log plot of grid convergence
plt.figure(figsize=(8,6))
plt.grid(True)
plt.xlabel('$\Delta t$', fontsize=18)
plt.ylabel('$L_1$-norm of the grid differences', fontsize=18)
plt.axis('equal')
plt.loglog(dt_values[:-1], diffgrid[:-1], color='k', ls='-',\
		lw=2, marker='o')

# ------- determining the order of convergence -------

r = 2
h = 0.001

dt_values2 = np.array([h,r*h,r**2])

u_values2 = np.empty_like(dt_values2, dtype=np.ndarray)

diffgrid2 = np.empty(2)

for i, dt in enumerate(dt_values2):
	N = int(T/dt) 	# number of time steps

	# discretize time-step
	t = np.linspace(0.0,T,N)

	# initialize the array for the soln
	u = np.empty((N,4))
	u[0] = np.array ([v0,theta0,x0,y0])

	#time loop
	for n in range(N-1):
		u[n+1] = euler_step(u[n], f, dt)

	#store value in grid
	u_values2[i] = u

# calculate f2-f1
diffgrid2[0] = get_diffgrid(u_values2[1], u_values2[0],\
							dt_values2[1])
# calculate f3-f2
diffgrid2[1] = get_diffgrid(u_values2[2], u_values2[1],\
							dt_values[2])
# finally calculating the order of convergence
p = (log(diffgrid2[1]) - log(diffgrid2[0]))/log(r)

print('The order of convergence is p = {:.3f}'.format(p))

plt.show()