# Module 1 Lesson 2 
# Phugoid Oscillation
# Ian Carr
# Sep 7 2014

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# time and space variables
T = 100.0
dt = 0.01
N = int(T/dt)+1
t = np.linspace(0.0,T,N) # cartesean grid

# ------- solving -------

# using eulers method to solve the 1st order ODEs

# initial conditions
z0 = 100. # altitude
v = 2.5 # upward velocity resulting from gust
zt = 100. #trim velocity
g = 9.81

u = np.array([z0,v])

# initialize an array to hold the changing angle values
z = np.zeros(N)
z[0] = z0 # setting first value

# time-loop using eulers method
for n in range(1,N):
    u = u + dt*np.array([u[1], g*(1-u[0]/zt)])
    z[n] = u[0]
    
# -------- plotting --------

plt.figure(figsize=(10,4)) # set plot size
plt.ylim(40,160)            # set y axis
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('t', fontsize=14) #x label
plt.ylabel('z', fontsize=14) #y label
plt.plot(t,z,'k-')
plt.show()
