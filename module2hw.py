# program to compute sympy equation for hw2
# Ian Carr

import numpy as np
import sympy as sp 
from math import *

from sympy.utilities.lambdify import lambdify

x = sp.symbols('x')
expr = sp.exp((sp.cos(x)**2*sp.sin(x)**3)/(4*x**5*2.71828**(x)))

print expr

exprprime = expr.diff(x)

f = lambdify(x, exprprime)
ans = f(2.2)

print ans
