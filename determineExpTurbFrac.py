
import numpy as np
import os
from scipy.optimize import curve_fit

actCrit = 1.388

def power_law(x, a, b):
    return a * np.power(x, b)

data = np.genfromtxt('dataAveraged.dat')

distToCrit = data[:,3] - actCrit
turbFrac =  data[:,8]

distToCritMin = 0.0005
distToCritMax = 0.01

popt, pcov = curve_fit(power_law,distToCrit[(distToCrit > distToCritMin) & (distToCrit < distToCritMax)],turbFrac[(distToCrit > distToCritMin) & (distToCrit < distToCritMax)], p0=[1.0, 0.34])

print()
print('exponent')
print(popt[1])
print(2.0*np.sqrt(np.diag(pcov))[1])




