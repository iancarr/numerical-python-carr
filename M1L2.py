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
v = 10 # upward velocity resulting from gust
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


# --------- computing exact solution ---------

# exact equation
z_exact = v*(zt/g)**.5*np.sin((g/zt)**.5*t)+\
            (z0-zt)*np.cos((g/zt)**.5*t)+zt
            
# plotting the exact soln
plt.figure(figsize=(10,4))
plt.ylim(40,160)
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('t',fontsize=14)
plt.ylabel('z',fontsize=14)
plt.plot(t,z)
plt.plot(t,z_exact)
plt.legend(['Numerical Solution','Analytical Solution']);

# ---------- comparing the solns with the norm ---------

# time-increment array
dt_values = np.array([0.1, 0.05, 0.01, 0.005,0.001,0.0001])

# array that will contain solution of each grid
z_values = np.empty_like(dt_values, dtype=np.ndarray)

for i, dt in enumerate(dt_values):
    N = int(T/dt)+1 # time steps
    t = np.linspace(0.0,T,N) # discreteize the time using linspace
    
    #IC
    u = np.array([z0,v])
    z = np.empty_like(t)
    z[0] = z0
    
    # time loop using euler's method
    for n in range(1,N):
        u = u + dt*np.array([u[1],g*(1-u[0]/zt)])
        z[n] = u[0]     # storing the elevation for the next step
        
    z_values[i] = z.copy()  # store the total elevation calculation grid i
    

def get_error(z,dt):
    
    # z: array of float, numerical solution
    # dt: float, time increment
    
    #returns err: float
    
    N = len(z)
    t = np.linspace(0.0,T,N)
    z_exact = v*(zt/g)**.5*np.sin((g/zt)**.5*t)+\
                (z0-zt)*np.cos((g/zt)**.5*t)+zt
    return dt * np.sum(np.abs(z-z_exact))

error_values = np.empty_like(dt_values)

# looping to fill error array
for i, dt in enumerate(dt_values):
    error_values[i] = get_error(z_values[i], dt)
    
# plotting error values
plt.figure(figsize=(10,6))
plt.tick_params(axis='both',labelsize=14)
plt.grid(True)
plt.xlabel('$\Delta t$', fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.loglog(dt_values, error_values,'ko-')
plt.axis('equal')
plt.show
print 'hellos'