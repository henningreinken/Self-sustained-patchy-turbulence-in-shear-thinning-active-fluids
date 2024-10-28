
import numpy as np
import os
import shutil
import subprocess
import time
import os.path
import sys

# get current directory
thisPath = os.getcwd()


# coefficients
visc0 = 1.5
shearRateRef = 2.0
exponent = 4.0


actStart1 = 1.37
actEnd1 = 1.55
actStep1 = 0.01

# actStart2 = 1.39
# actEnd2 = 1.399
# actStep2 = 0.001

# actStart3 = 1.3868
# actEnd3 = 1.3898
# actStep3 = 0.0002

# actStart4 = 1.325
# actEnd4 = 1.385
# actStep4 = 0.005

numAct1 = int(np.round((actEnd1 - actStart1)/actStep1 + 1))
actArray1 = np.linspace(actStart1,actEnd1,numAct1,endpoint=True)

# numAct2 = int(np.round((actEnd2 - actStart2)/actStep2 + 1))
# actArray2 = np.linspace(actStart2,actEnd2,numAct2,endpoint=True)

# numAct3 = int(np.round((actEnd3 - actStart3)/actStep3 + 1))
# actArray3 = np.linspace(actStart3,actEnd3,numAct3,endpoint=True)

# numAct4 = int(np.round((actEnd4 - actStart4)/actStep4 + 1))
# actArray4 = np.linspace(actStart4,actEnd4,numAct4,endpoint=True)

# actArray = np.concatenate((actArray4,actArray3,actArray2, actArray1), axis=0)

actArray = actArray1

print(actArray)

numFilesAv = 2

averagedDataFileName = str(thisPath) + '/dataAveraged.dat'

averagedDataFile = open(averagedDataFileName,'w')
averagedDataFile.close()

averagedDataFile = open(averagedDataFileName,'a')

# walk through all the values
for act in actArray:

	print('act = ' + str(np.round(act,5)))

	# folder name
	jobName = 'a%08.5f'%(act)
	jobName = jobName.replace('.','d',1)
	folderName = thisPath + "/" + jobName

	nAv = 0.0
	dataAv = np.zeros(10)
	
	runStart = 1
	runEnd = 0

	for runNum in range(1,30):
		dataFileName = str(folderName) + '/dataMean%04d.dat'%(runNum)
		if os.path.isfile(dataFileName):
			runEnd = runNum
			
	if runEnd > runStart + numFilesAv - 1:
		runStart = runEnd - numFilesAv + 1
			
	print('average from run ' + str(runStart) + ' to ' + str(runEnd))
	print()
			
	for runNum in range(runStart,runEnd+1):	
		dataFileName = str(folderName) + '/dataMean%04d.dat'%(runNum)
		dataTemp = np.genfromtxt(dataFileName)
		sizeData = np.shape(dataTemp)[0]
		for i in range(0,sizeData):
			nAv = nAv + 1.0
			dataAv[0] = dataAv[0] + dataTemp[i,1] # mean enstrophy
			dataAv[1] = dataAv[1] + dataTemp[i,2] # vrms
			dataAv[2] = dataAv[2] + dataTemp[i,3] # mean viscosity
			dataAv[3] = dataAv[3] + dataTemp[i,4] # mean shear rate
			dataAv[4] = dataAv[4] + dataTemp[i,5] # area fraction turbulence

	dataAv = dataAv/nAv

	meanEnstr = dataAv[0]
	varEnstr = 0.0
	meanTurbFrac = dataAv[4]
	varTurbFrac = 0.0
	for runNum in range(runStart,runEnd+1):	
		dataFileName = str(folderName) + '/dataMean%04d.dat'%(runNum)
		dataTemp = np.genfromtxt(dataFileName)
		sizeData = np.shape(dataTemp)[0]
		enstrTemp = 0.0
		turbFracTemp = 0.0
		for i in range(0,sizeData):
			enstrTemp = enstrTemp + dataTemp[i,1]/float(sizeData)
			turbFracTemp = turbFracTemp + dataTemp[i,5]/float(sizeData)
		# print(turbFracTemp)
		varEnstr = varEnstr + (enstrTemp - meanEnstr)**2/float(runEnd - runStart)
		varTurbFrac = varTurbFrac + (turbFracTemp - meanTurbFrac)**2/float(runEnd - runStart)
	stDevEnstr = np.sqrt(varEnstr)
	stErrEnstr = stDevEnstr/np.sqrt(float(runEnd - runStart))
	stDevTurbFrac = np.sqrt(varTurbFrac)
	stErrTurbFrac = stDevTurbFrac/np.sqrt(float(runEnd - runStart))
	# print(stErrTurbFrac)
	# print()

	averagedDataFile.write(str(shearRateRef) + ' ' + str(visc0)+ ' ' + str(exponent) + ' ' + str(act) + ' ' + str(dataAv[0]) + ' ' + str(dataAv[1]) + ' ' + str(dataAv[2]) + ' ' + str(dataAv[3]) + ' ' + str(dataAv[4]) + ' ' + str(stErrTurbFrac) + ' ' + str(stErrEnstr) + ' \n')

averagedDataFile.close()

