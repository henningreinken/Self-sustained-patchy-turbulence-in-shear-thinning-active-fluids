
import numpy as np
import os
import shutil
import subprocess
import time
import os.path
import sys

# get current directory
thisPath = os.getcwd()


shearRateRef = 2.0
visc0 = 1.5
exponent = 4.0


numFilesAv = 3

actStart1 = 1.4
actEnd1 = 1.68
actStep1 = 0.01

actStart2= 1.387
actEnd2 = 1.399
actStep2 = 0.001


numAct1 = int(np.round((actEnd1 - actStart1)/actStep1 + 1))
actArray1 = np.linspace(actStart1,actEnd1,numAct1,endpoint=True)

numAct2 = int(np.round((actEnd2 - actStart2)/actStep2 + 1))
actArray2 = np.linspace(actStart2,actEnd2,numAct2,endpoint=True)


actArray = np.concatenate((actArray2, actArray1), axis=0)

averagedDataFileName = str(thisPath) + '/dataKurtosis.dat'

averagedDataFile = open(averagedDataFileName,'w')
averagedDataFile.close()

averagedDataFile = open(averagedDataFileName,'a')

for act in actArray:

	jobName = 'a%08.5f'%(act)
	jobName = jobName.replace('.','d',1)
	folderName = thisPath + "/" + jobName
	
	
	runStart = 1
	runEnd = 0
	
	for runNum in range(1,10):
		dataFileName = str(folderName) + '/dataDistrv%04d.dat'%(runNum)
		if os.path.isfile(dataFileName):
			runEnd = runNum
			
	if runEnd > runStart + numFilesAv - 1:
		runStart = runEnd - numFilesAv + 1
	
	
	print(act)		
	print(runStart)
	print(runEnd)

	dv = 0.1
	vmax = 200.0
	numDistrv = round(vmax/dv)
	vArray = np.linspace(0.5*dv,vmax+0.5*dv,numDistrv,endpoint=False)

	vArrayPM = np.append(-np.flip(vArray,axis=0),vArray)

	stDev = 0.0
	kurtAv = 0.0
	vArrayAv = np.zeros(2*numDistrv)
	vDistrAv = np.zeros(2*numDistrv)
	nAv = 0.0

	for runNum in range(runStart,runEnd+1):
		
		dataFileNameTemp = folderName+'/dataDistrv%04d.dat'%(runNum)
		
		if os.path.isfile(dataFileNameTemp) == True:
		
			dataTemp = np.genfromtxt(folderName+'/dataDistrv%04d.dat'%(runNum))
			
			distrvxPM = 0.5*np.append(np.flip(dataTemp[1,:],axis=0),dataTemp[0,:])
			distrvyPM = 0.5*np.append(np.flip(dataTemp[3,:],axis=0),dataTemp[2,:])
			
			stDevx = np.sqrt(np.sum(vArrayPM*vArrayPM*distrvxPM*dv)) 
			stDevy = np.sqrt(np.sum(vArrayPM*vArrayPM*distrvyPM*dv)) 
			stDevAv = 0.5*(stDevx + stDevy)

			kurtx = np.sum(vArrayPM*vArrayPM*vArrayPM*vArrayPM*distrvxPM*dv)/stDevx/stDevx/stDevx/stDevx
			kurty = np.sum(vArrayPM*vArrayPM*vArrayPM*vArrayPM*distrvyPM*dv)/stDevy/stDevy/stDevy/stDevy
			kurtAv = kurtAv + 0.5*(kurtx + kurty)
					
			vArrayPMnorm = vArrayPM/stDevAv	
			
			stDev = stDev + stDevAv
			vArrayAv = vArrayAv + vArrayPM/stDevAv	
			vDistrAv = vDistrAv + 0.5*(distrvxPM + distrvyPM)*stDevAv
			nAv = nAv + 1.0
	
	if nAv > 0:
		stDev = stDev/nAv
		kurtAv = kurtAv/nAv
		vArrayAv = vArrayAv/nAv
		vDistrAv = vDistrAv/nAv

	print(stDev)
	print(kurtAv)
	print()
	
	averagedDataFile.write(str(shearRateRef) + ' ' + str(visc0)+ ' ' + str(exponent) + ' ' + str(act) + ' ' + str(stDev) + ' ' + str(kurtAv) + ' \n')

averagedDataFile.close()




