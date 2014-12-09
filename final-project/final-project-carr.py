# Final Project - Ian Carr
# Blasius solution - introduction to fluids

import numpy as np 
import matplotlib.pyplot as plt 

# building initial parameters
nfinal = 5.	# final value of n
dn = 0.01 	# step size
N = int(nfinal/dn)
n = np.linspace(0.0,nfinal,N)

f = np.zeros(N)
f1 = np.zeros(N)
f2 = np.zeros(N)

# initial conditions
f[0] = 0. 
f1[0] = 0.

# our shot for the value of f at infty
f2shot = np.linspace(0.3,1.0,1000)

for i in range(len(f2shot)):
	f2[0] = f2shot[i]
	# iterating using euler's method
	for i in range(0,N-1):
		f[i+1] = f[i] + f1[i]*dn
		f1[i+1] = f1[i] + f2[i]*dn
		f2[i+1] = f2[i] - 0.5*f[i]*f2[i]*dn
		if f1[-1] > 1:
			break

print f1[-1]
print f2[-1]

# plotting
plt.figure()
plt.plot(f,n)
plt.xlim(0,1.2)
plt.ylim(0,12)
plt.ylabel('$\eta$',fontsize=18)
plt.xlabel('$u/u_\infty$',fontsize=18)
plt.show()
