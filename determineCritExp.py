
import numpy as np
import os
from scipy.optimize import curve_fit

# get current directory
thisPath = os.getcwd()

N = 64
L = 2.0*np.pi*12.0
# N = 768
# L = 2.0*np.pi*64.0
dx = L/float(N)

# coefficients
act = 1.39
visc0 = 1.5
shearRateRef = 2.0
exponent = 4.0

dt = 0.5
numT = 5000

lowerLimitSpace = 20
upperLimitSpace = 30

lowerLimitTime = 10
upperLimitTime = 100

# where to save everything
jobName = 'a%08.5f'%(act)
jobName = jobName.replace('.','d',1)
folderName = thisPath + "/" + jobName

def power_law(x, a, b):
    return a * np.power(x, -b)

dataFileName = folderName + '/distributionGapLength.dat'
data = np.genfromtxt(dataFileName)
sizesArr= np.linspace(dx,L,num=N,endpoint=True)

popt, pcov = curve_fit(power_law,sizesArr[(sizesArr > lowerLimitSpace) & (sizesArr < upperLimitSpace)],data[(sizesArr > lowerLimitSpace) & (sizesArr < upperLimitSpace),1], p0=[0.001, 1.2])

print()
print('spatial exponent')
print(popt[1])
print('uncertainty')
print(2.0*np.sqrt(np.diag(pcov))[1])


dataFileName = folderName + '/distributionQuiescentDuration.dat'
data = np.genfromtxt(dataFileName)

sizesArr = np.linspace(dt,numT*dt,num=numT,endpoint=True)

popt, pcov = curve_fit(power_law,sizesArr[(sizesArr > lowerLimitTime) & (sizesArr < upperLimitTime)],data[(sizesArr > lowerLimitTime) & (sizesArr < upperLimitTime),1], p0=[0.001, 1.2])

print()
print('temporal exponent')
print(popt[1])
print('uncertainty')
print(2.0*np.sqrt(np.diag(pcov))[1])
print()
