# Final Project - Ian Carr
# Blasius solution - introduction to fluids

import numpy as np 
import matplotlib.pyplot as plt 

def blasius(eta_max, steps, fppwall):
	"""
	Parameters:
	eta_max - float - max eta for calculation
	steps - int - num steps between 0 and eta_max
	fppwall - float - initial value of 2nd derivative

	Returns:
	eta - the similarity coordinate normal to the wall
	f, fp, fpp, fppp - blasius function and first 3 derivatives
	"""

	deta = eta_max/(steps-1)

	eta = np.zeros(steps)
	f = np.zeros_like(eta)
	fp = np.zeros_like(eta)
	fpp = np.zeros_like(eta)
	fppp = np.zeros_like(eta)

	# initial guess for fpp
	fpp(0) = fppwall

	for i in range(steps-1):
		eta[i+1] = eta[i] + deta
		
		# predictor
		fbar = f[i] + deta
		fpbar = fp[i] + deta*fp[i]
		fppbar = fpp[i] + deta*fpp[i]
		fpppbar = -fbar*fppbar/2

		# corrector
		f[i+1] = f[i] + deta*(fp[i] + fpbar)/2
		fp[i+1] = f[i] + deta*(fpp[i]+fppbar)/2
		fpp[i+1] = fpp[i] + deta*(fppp[i] + fpppbar)/2
		fppp[i+1] = -f[i+1]*fpp[i+1]/2

	return eta, f, fp, fpp, fppp


