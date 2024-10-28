
import numpy as np
import os

# get current directory
thisPath = os.getcwd()

# N = 768
# sizeInWavelength = 64.0
N = 64
sizeInWavelength = 12.0
L = sizeInWavelength*2.0*np.pi
dx = L/N

# coefficients
act = 1.39
visc0 = 1.5
shearRateRef = 2.0
exponent = 4.0

# where to save everything
jobName = 'a%08.5f'%(act)
jobName = jobName.replace('.','d',1)
folderName = thisPath + "/" + jobName

runNumStart = 4
runNumEnd = 5

runNumArray = np.arange(runNumStart, runNumEnd+1, dtype=int)

data = np.zeros((4,N))
nAv = 0.0

for runNum in runNumArray:
	
	dataFile = folderName+'/clusterSizeDirDistr%04d.dat'%(runNum)
	data = data + np.genfromtxt(dataFile)
	nAv = nAv + 1.0

data = data/nAv

dataAv = 0.5*(data[2,:] + data[3,:])

# normalize
dataAv = dataAv/dataAv[0]

sizesArr = np.linspace(dx,L,num=N,endpoint=True)

np.savetxt(folderName+'/distributionGapLength.dat',np.transpose(np.stack((sizesArr,dataAv))))



numSteps = 5000
dt = 0.5

data = np.zeros((2,numSteps))
nAv = 0.0

for runNum in runNumArray:
	
	dataFile = folderName+'/durationDistr%04d.dat'%(runNum)
	data = data + np.genfromtxt(dataFile)
	nAv = nAv + 1.0

data = data/nAv

dataAv = data[1,:]

dataAv = dataAv/dataAv[0]

sizesArr = np.linspace(dt,dt*numSteps,num=numSteps,endpoint=True)

np.savetxt(folderName+'/distributionQuiescentDuration.dat',np.transpose(np.stack((sizesArr,dataAv))))





