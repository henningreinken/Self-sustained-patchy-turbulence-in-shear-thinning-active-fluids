
import os
import numpy as np
import sys
import time
from scipy.ndimage import label

# get current directory
thisPath = os.getcwd()

# coefficients
act = 1.39
visc0 = 1.5
shearRateRef = 2.0
exponent = 4.0

# numerical parameters

dt = 0.05
# N = 768
# sizeInWavelength = 64.0
N = 64
sizeInWavelength = 12.0
L = sizeInWavelength*2.0*np.pi
dx = L/N
nStep = 0
# tEnd = 1000.0
tEnd = 200.0

numSteps = round(tEnd/dt) 

# where to save everything
jobName = 'a%08.5f'%(act)
jobName = jobName.replace('.','d',1)
folderName = thisPath + "/" + jobName

runNum = 0
for i in range(1,100):
	if os.path.isfile(folderName+'/datavxInit%04d.dat'%i):
		runNum = i

dtPrint = 10.0
dnPrint = round(dtPrint/dt)

dtAna = 0.5
dnAna = round(dtAna/dt)
sizeAna = round(tEnd/dtAna)
nAna = 0
nAnaDuration = 0.0

# to check if parameters are forwarded correctly
print('activity strength = ' + str(act), flush=True)
print('reference shear rate = ' + str(shearRateRef), flush=True)
print('zero shear viscosity = ' + str(visc0), flush=True)
print('exponent = ' + str(exponent), flush=True)
print('grid points = ' + str(N) + 'x' + str(N), flush=True)
print('box size = ' + str(L), flush=True)
print('run ' + str(runNum), flush=True)
print()

# definition of the wavevector
k = 1j*np.fft.fftfreq(N,d=L/2.0/np.pi/N)
kx, ky = np.meshgrid(k, k, sparse=False, indexing='ij')

# for plotting: numbering meshgrid
nx, ny = np.meshgrid(np.linspace(0,N,num=N,endpoint=False),np.linspace(0,N,num=N,endpoint=False),indexing='ij')

# mask for dealiaising
dealiasingFrac = 2.0/6.0
dealiasingMask = np.ones((N,N))
dealiasingMask[:,round(N/2)-round(N/2*dealiasingFrac):round(N/2)+round(N/2*dealiasingFrac)] = 0
dealiasingMask[round(N/2)-round(N/2*dealiasingFrac):round(N/2)+round(N/2*dealiasingFrac),:] = 0

# Laplacian and biharmonic derivative in Fourier space
kLaplacian = kx*kx + ky*ky
kLaplacianPoisson = kLaplacian
kLaplacianPoisson = 1j*np.zeros((N,N))
kLaplacianPoisson[:,:] = kLaplacian[:,:]
kLaplacianPoisson[0,0] = 0.000000001 # to avoid divide-by-zero
kBiharmonic = kLaplacian*kLaplacian

# random initial values
initVarv = 0.01
vx = -initVarv + 2.0*initVarv*np.random.rand(N,N)
vy = -initVarv + 2.0*initVarv*np.random.rand(N,N)

# scaling such that mean velocity is zero
vx = vx - vx.mean()
vy = vy - vy.mean()

# pressure correction
vxF = np.fft.fft2(vx)
vyF = np.fft.fft2(vy)
vDivF = kx*vxF + ky*vyF
pvF = vDivF/kLaplacianPoisson
pv_xF = kx*pvF
pv_yF = ky*pvF
vxNewF = vxF - pv_xF
vyNewF = vyF - pv_yF

# initial values: data from data files
if runNum != 0:
	vx = np.genfromtxt(folderName+'/datavxInit%04d.dat'%(runNum))
	vy = np.genfromtxt(folderName+'/datavyInit%04d.dat'%(runNum))

# to estimate remaining time
timeSinceLastPrint = time.time()

# coarse-grained state by me
# parameters and quantities for coarse-graining
NCG = int(sizeInWavelength)
dPointsCG = int(N/NCG)
nExtend = dPointsCG + 1
viscExtended = np.zeros((N+2*nExtend,N+2*nExtend))
vAbsExtended = np.zeros((N+2*nExtend,N+2*nExtend))
enstrExtended = np.zeros((N+2*nExtend,N+2*nExtend))

viscCG = np.zeros((N,N))

# extends the field to a larger area to account for periodic boundaries
# and increase the accuracy of the interpolation
def calcExtendedField(fieldIn):
	fieldExtendedOut = np.zeros((N+2*nExtend,N+2*nExtend))
	for i in range(0,N+2*nExtend):
		for j in range(0,N+2*nExtend):
			if i < nExtend:
				if j < nExtend:
					fieldExtendedOut[i,j] = fieldIn[N-nExtend+i,N-nExtend+j]
				elif j < (N+nExtend):
					fieldExtendedOut[i,j] = fieldIn[N-nExtend+i,j-nExtend]
				else:
					fieldExtendedOut[i,j] = fieldIn[N-nExtend+i,j-N-nExtend]
			elif i < (N+nExtend):
				if j < nExtend:
					fieldExtendedOut[i,j] = fieldIn[i-nExtend,N-nExtend+j]
				elif j < (N+nExtend):
					fieldExtendedOut[i,j] = fieldIn[i-nExtend,j-nExtend]
				else:
					fieldExtendedOut[i,j] = fieldIn[i-nExtend,j-N-nExtend]
			else:
				if j < nExtend:
					fieldExtendedOut[i,j] = fieldIn[i-N-nExtend,N-nExtend+j]
				elif j < (N+nExtend):
					fieldExtendedOut[i,j] = fieldIn[i-N-nExtend,j-nExtend]
				else:
					fieldExtendedOut[i,j] = fieldIn[i-N-nExtend,j-N-nExtend]
	return fieldExtendedOut

# cluster size distributions (spatially)
clusterTurbSizeDistr = np.zeros(N*N)
clusterQuieSizeDistr = np.zeros(N*N)

clusterQuieSizeXDistr = np.zeros(N)
clusterTurbSizeXDistr = np.zeros(N)
clusterQuieSizeYDistr = np.zeros(N)
clusterTurbSizeYDistr = np.zeros(N)

# duration distribution
durationTurbDistr = np.zeros(5000)
durationQuieDistr = np.zeros(5000)

# matrix tracking how long a grid point is turbulent/quiescent
if os.path.isfile(folderName+'/dataStateCounter%04d.dat'%runNum):
	stateCounter = np.genfromtxt(folderName+'/dataStateCounter%04d.dat'%runNum)
else:
	stateCounter = np.zeros((N,N))
	

# linear operators
LINvF = kLaplacian + act*(2.0*kLaplacian*kLaplacian + kLaplacian*kLaplacian*kLaplacian)

# right-hand side, needed for calculation of Runge-Kutta steps
def rhs(dt,vx,vy):

	vxF = np.fft.fft2(vx)
	vyF = np.fft.fft2(vy)

	# dealiasing
	vxF = vxF*dealiasingMask
	vyF = vyF*dealiasingMask

	# velocity gradients
	vx_xF = kx*vxF
	vx_yF = ky*vxF
	vy_xF = kx*vyF
	vy_yF = ky*vyF
	vx_x = np.real(np.fft.ifft2(vx_xF))
	vx_y = np.real(np.fft.ifft2(vx_yF))
	vy_x = np.real(np.fft.ifft2(vy_xF))
	vy_y = np.real(np.fft.ifft2(vy_yF))

	# advection term
	advx = vx*vx_x + vy*vx_y
	advy = vx*vy_x + vy*vy_y
	advxF = np.fft.fft2(advx)
	advyF = np.fft.fft2(advy)

	# local shear rate and viscosity
	shearRate = np.power(2.0*vx_x*vx_x + (vx_y + vy_x)*(vx_y + vy_x) + 2.0*vy_y*vy_y,0.5)
	viscAdd = (visc0 - 1.0)/(1.0 + np.power(shearRate/shearRateRef,exponent))

	viscStrxx = 2.0*viscAdd*vx_x
	viscStrxy = viscAdd*(vx_y + vy_x)
	viscStryy = 2.0*viscAdd*vy_y

	viscStrxxF = np.fft.fft2(viscStrxx)
	viscStrxyF = np.fft.fft2(viscStrxy)
	viscStryyF = np.fft.fft2(viscStryy)

	# time propagation via operator splitting (linear and nonlinear)
	vxNewF = np.exp(dt*LINvF)*(vxF + dt*(kx*viscStrxxF + ky*viscStrxyF - advxF))
	vyNewF = np.exp(dt*LINvF)*(vyF + dt*(kx*viscStrxyF + ky*viscStryyF - advyF))

	# pressure correction
	vDivF = kx*vxNewF + ky*vyNewF
	pvF = vDivF/kLaplacianPoisson
	pv_xF = kx*pvF
	pv_yF = ky*pvF
	vxNewF = vxNewF - pv_xF
	vyNewF = vyNewF - pv_yF

	# back to real space
	vxNew = np.real(np.fft.ifft2(vxNewF))
	vyNew = np.real(np.fft.ifft2(vyNewF))

	RHSvx = vxNew - vx
	RHSvy = vyNew - vy

	return RHSvx, RHSvy


for nStep in range(0,numSteps):
	
	t = round(nStep*dt,6)
	
	# show progress
	if nStep%dnPrint == 0:
		vxF = np.fft.fft2(vx)
		vyF = np.fft.fft2(vy)
		vxF = vxF*dealiasingMask
		vyF = vyF*dealiasingMask
		vx_xF = kx*vxF
		vx_yF = ky*vxF
		vy_xF = kx*vyF
		vy_yF = ky*vyF
		vx_x = np.real(np.fft.ifft2(vx_xF))
		vx_y = np.real(np.fft.ifft2(vx_yF))
		vy_x = np.real(np.fft.ifft2(vy_xF))
		vy_y = np.real(np.fft.ifft2(vy_yF))
		vVortF = kx*vyF - ky*vxF
		vVort = np.real(np.fft.ifft2(vVortF))
		shearRate = np.power(2.0*vx_x*vx_x + (vx_y + vy_x)*(vx_y + vy_x) + 2.0*vy_y*vy_y,0.5)
		visc = 1.0 + (visc0 - 1.0)/(1.0 + np.power(shearRate/shearRateRef,exponent))
		print('t = ' + str(t), flush=True)
		print('maximum vorticity in v = ' + str(0.5*(np.max(vVort)-np.min(vVort))), flush=True)
		print('maximum value of v = ' + str(np.max(vx*vx+vy*vy)), flush=True)
		print('mean value of v squared = ' + str(np.mean(vx*vx+vy*vy)), flush=True)
		print('maximum divergence in v = ' + str(np.max(np.abs(np.real(np.fft.ifft2(kx*vxF + ky*vyF))))), flush=True)
		print('maximum local shear rate = ' + str(np.max(shearRate)), flush=True)
		print('maximum local viscosity = ' + str(np.max(visc)), flush=True)
		print('mean local viscosity = ' + str(np.mean(visc)), flush=True)
		print('time remaining: ' + str((time.time() - timeSinceLastPrint)/60/60*(tEnd-t)/dtPrint) + ' hours', flush=True)
		print('vx = ' + str(np.mean(vx)), flush=True)
		print('vy = ' + str(np.mean(vy)), flush=True)
		print()
		timeSinceLastPrint = time.time()

	# Runge-Kutta intermediate steps
	k1vx, k1vy = rhs(dt,vx,vy)
	k2vx, k2vy = rhs(dt,vx + 0.5*k1vx,vy + 0.5*k1vy)
	k3vx, k3vy = rhs(dt,vx + 0.5*k2vx,vy + 0.5*k2vy)
	k4vx, k4vy = rhs(dt,vx + k3vx,vy + k3vy)

	# calculate new velocity field at the next step
	vx = vx + (k1vx + 2.0*k2vx + 2.0*k3vx + k4vx)/6.0
	vy = vy + (k1vy + 2.0*k2vy + 2.0*k3vy + k4vy)/6.0
		
	# calculate mean values
	if nStep%dnAna == 0:
		vxF = np.fft.fft2(vx)
		vyF = np.fft.fft2(vy)
		vxF = vxF*dealiasingMask
		vyF = vyF*dealiasingMask
		vx_xF = kx*vxF
		vx_yF = ky*vxF
		vy_xF = kx*vyF
		vy_yF = ky*vyF
		vx_x = np.real(np.fft.ifft2(vx_xF))
		vx_y = np.real(np.fft.ifft2(vx_yF))
		vy_x = np.real(np.fft.ifft2(vy_xF))
		vy_y = np.real(np.fft.ifft2(vy_yF))
		vVortF = kx*vyF - ky*vxF
		vVort = np.real(np.fft.ifft2(vVortF))
		shearRate = np.power(2.0*vx_x*vx_x + (vx_y + vy_x)*(vx_y + vy_x) + 2.0*vy_y*vy_y,0.5)
		visc = 1.0 + (visc0 - 1.0)/(1.0 + np.power(shearRate/shearRateRef,exponent))
		
		viscExtended = calcExtendedField(visc)
		
		for i in range(0,N):
			for j in range(0,N):
				iStart = round(nExtend + i - dPointsCG/2)
				iEnd = round(nExtend + i + dPointsCG/2)
				jStart = round(nExtend + j - dPointsCG/2)
				jEnd = round(nExtend + j + dPointsCG/2)
				viscCG[i,j] = np.mean(viscExtended[iStart:iEnd,jStart:jEnd])
		
		# create matrix with 0 for quiescence and 1 for turbulence
		turbYesNo = np.zeros((N,N))
		for i in range(0,N):
			for j in range(0,N):
				if viscCG[i,j] < act:
					turbYesNo[i,j] = 1
		
		# label turbulent clusters
		turbYesNoLabeled, numClusters = label(turbYesNo)
		
		# account for periodic boundary conditions
		for i in range(0,N):
			if turbYesNoLabeled[i,0] > 0 and turbYesNoLabeled[i,-1] > 0:
				turbYesNoLabeled[turbYesNoLabeled == turbYesNoLabeled[i,-1]] = turbYesNoLabeled[i, 0]
		for j in range(0,N):
			if turbYesNoLabeled[0,j] > 0 and turbYesNoLabeled[-1,j] > 0:
				turbYesNoLabeled[turbYesNoLabeled == turbYesNoLabeled[-1,j]] = turbYesNoLabeled[0,j]
		
		# count clusters of particular sizes
		for n in range(0,np.max(turbYesNoLabeled)):
			clusterSize = np.sum(turbYesNo[turbYesNoLabeled[:,:]==(n+1)])
			if clusterSize > 0:
				clusterTurbSizeDistr[int(clusterSize-1)] += 1.0
					
		
		for i in range(0,N):
			turbYesNoSlice = turbYesNo[i,:]
			if turbYesNoSlice[0] == 0 and turbYesNoSlice[N-1] == 0:
				jStart = 1
				jEnd = N - 2
				lGap = 2
				while turbYesNoSlice[jStart] == 0 and jStart < N - 1:
					lGap += 1
					jStart += 1
				if jStart == N - 1:
					clusterQuieSizeYDistr[int(N-1)] += 1.0
				else:
					while turbYesNoSlice[jEnd] == 0:
						lGap += 1
						jEnd -= 1
					clusterQuieSizeYDistr[int(lGap-1)] += 1.0
					lGap = 1
					for j in range(jStart+1,jEnd+2):
						if turbYesNoSlice[j] == 0:
							if lGap > 0:
								clusterTurbSizeYDistr[int(lGap-1)] += 1.0
								lGap = 0
							lGap -= 1
						else:
							if lGap < 0:
								clusterQuieSizeYDistr[int(-lGap-1)] += 1.0
								lGap = 0
							lGap += 1
			elif turbYesNoSlice[0] == 1 and turbYesNoSlice[N-1] == 1:
				jStart = 1
				jEnd = N - 2
				lGap = 2
				while turbYesNoSlice[jStart] == 1 and jStart < N - 1:
					lGap += 1
					jStart += 1
				if jStart == N - 1:
					clusterTurbSizeYDistr[int(N-1)] += 1.0
				else:
					while turbYesNoSlice[jEnd] == 1:
						lGap += 1
						jEnd -= 1
					clusterTurbSizeYDistr[int(lGap-1)] += 1.0
					lGap = -1
					for j in range(jStart+1,jEnd+2):
						if turbYesNoSlice[j] == 0:
							if lGap > 0:
								clusterTurbSizeYDistr[int(lGap-1)] += 1.0
								lGap = 0
							lGap -= 1
						else:
							if lGap < 0:
								clusterQuieSizeYDistr[int(-lGap-1)] += 1.0
								lGap = 0
							lGap += 1
			else:
				if turbYesNoSlice[0] == 0:
					lGap = -1
				else:
					lGap = 1
				for j in range(1,N):
					if turbYesNoSlice[j] == 0:
						if lGap > 0:
							clusterTurbSizeYDistr[int(lGap-1)] += 1.0
							lGap = 0
						lGap -= 1
					else:
						if lGap < 0:
							clusterQuieSizeYDistr[int(-lGap-1)] += 1.0
							lGap = 0
						lGap += 1
				if lGap > 0:
					clusterTurbSizeYDistr[int(lGap-1)] += 1.0
				else:
					clusterQuieSizeYDistr[int(-lGap-1)] += 1.0
					
		
		for i in range(0,N):
			turbYesNoSlice = turbYesNo[:,i]
			if turbYesNoSlice[0] == 0 and turbYesNoSlice[N-1] == 0:
				jStart = 1
				jEnd = N - 2
				lGap = 2
				while turbYesNoSlice[jStart] == 0 and jStart < N - 1:
					lGap += 1
					jStart += 1
				if jStart == N - 1:
					clusterQuieSizeXDistr[int(N-1)] += 1.0
				else:
					while turbYesNoSlice[jEnd] == 0:
						lGap += 1
						jEnd -= 1
					clusterQuieSizeXDistr[int(lGap-1)] += 1.0
					lGap = 1
					for j in range(jStart+1,jEnd+2):
						if turbYesNoSlice[j] == 0:
							if lGap > 0:
								clusterTurbSizeXDistr[int(lGap-1)] += 1.0
								lGap = 0
							lGap -= 1
						else:
							if lGap < 0:
								clusterQuieSizeXDistr[int(-lGap-1)] += 1.0
								lGap = 0
							lGap += 1
			elif turbYesNoSlice[0] == 1 and turbYesNoSlice[N-1] == 1:
				jStart = 1
				jEnd = N-2
				lGap = 2
				while turbYesNoSlice[jStart] == 1 and jStart < N - 1:
					lGap += 1
					jStart += 1
				if jStart == N - 1:
					clusterTurbSizeXDistr[int(N-1)] += 1.0
				else:
					while turbYesNoSlice[jEnd] == 1:
						lGap += 1
						jEnd -= 1
					clusterTurbSizeXDistr[int(lGap-1)] += 1.0
					lGap = -1
					for j in range(jStart+1,jEnd+2):
						if turbYesNoSlice[j] == 0:
							if lGap > 0:
								clusterTurbSizeXDistr[int(lGap-1)] += 1.0
								lGap = 0
							lGap -= 1
						else:
							if lGap < 0:
								clusterQuieSizeXDistr[int(-lGap-1)] += 1.0
								lGap = 0
							lGap += 1
			else:
				if turbYesNoSlice[0] == 0:
					lGap = -1
				else:
					lGap = 1
				for j in range(1,N):
					if turbYesNoSlice[j] == 0:
						if lGap > 0:
							clusterTurbSizeXDistr[int(lGap-1)] += 1.0
							lGap = 0
						lGap -= 1
					else:
						if lGap < 0:
							clusterQuieSizeXDistr[int(-lGap-1)] += 1.0
							lGap = 0
						lGap += 1
				if lGap > 0:
					clusterTurbSizeYDistr[int(lGap-1)] += 1.0
				else:
					clusterQuieSizeYDistr[int(-lGap-1)] += 1.0
		
		# label quiescent clusters
		turbYesNoReversed = - (turbYesNo - 1)
		quieYesNoLabeled, numClusters = label(turbYesNoReversed)
		
		# account for periodic boundary conditions
		for i in range(0,N):
			if quieYesNoLabeled[i,0] > 0 and quieYesNoLabeled[i,-1] > 0:
				quieYesNoLabeled[quieYesNoLabeled == quieYesNoLabeled[i,-1]] = quieYesNoLabeled[i, 0]
		for j in range(0,N):
			if quieYesNoLabeled[0,j] > 0 and quieYesNoLabeled[-1,j] > 0:
				quieYesNoLabeled[quieYesNoLabeled == quieYesNoLabeled[-1,j]] = quieYesNoLabeled[0,j]
				
		# count clusters of particular sizes
		for n in range(0,np.max(quieYesNoLabeled)):
			clusterSize = np.sum(turbYesNoReversed[quieYesNoLabeled[:,:]==(n+1)])
			if clusterSize > 0:
				clusterQuieSizeDistr[int(clusterSize-1)] += 1.0

		for i in range(0,N):
			for j in range(0,N):
				if viscCG[i,j] < act:
					if stateCounter[i,j] < 0:
						if (-stateCounter[i,j]-1) < 5000:
							durationQuieDistr[int(-stateCounter[i,j]-1)] += 1
						stateCounter[i,j] = 0
					stateCounter[i,j] += 1
				else:
					if stateCounter[i,j] > 0:
						if (stateCounter[i,j]-1) < 5000:
							durationTurbDistr[int(stateCounter[i,j]-1)] += 1
						stateCounter[i,j] = 0
					stateCounter[i,j] -= 1
		
		nAna = nAna + 1

# save cluster size distribution
clusterTurbSizeDistr = clusterTurbSizeDistr/float(nAna)/float(N*N)
clusterQuieSizeDistr = clusterQuieSizeDistr/float(nAna)/float(N*N)
np.savetxt(folderName+'/clusterSizeDistr%04d.dat'%(runNum),np.stack((clusterTurbSizeDistr,clusterQuieSizeDistr)))

clusterTurbSizeXDistr = clusterTurbSizeXDistr/float(nAna)/float(N*N)
clusterQuieSizeXDistr = clusterQuieSizeXDistr/float(nAna)/float(N*N)
clusterTurbSizeYDistr = clusterTurbSizeYDistr/float(nAna)/float(N*N)
clusterQuieSizeYDistr = clusterQuieSizeYDistr/float(nAna)/float(N*N)
np.savetxt(folderName+'/clusterSizeDirDistr%04d.dat'%(runNum),np.stack((clusterTurbSizeXDistr,clusterTurbSizeYDistr,clusterQuieSizeXDistr,clusterQuieSizeYDistr)))

# save cluster size distribution
durationTurbDistr = durationTurbDistr/float(N*N)
durationQuieDistr = durationQuieDistr/float(N*N)
np.savetxt(folderName+'/durationDistr%04d.dat'%(runNum),np.stack((durationTurbDistr,durationQuieDistr)))

# save last time step for the field for continuation of calculation
np.savetxt(folderName+'/datavxInit%04d.dat'%(runNum+1),vx)
np.savetxt(folderName+'/datavyInit%04d.dat'%(runNum+1),vy)

np.savetxt(folderName+'/dataStateCounter%04d.dat'%(runNum+1),stateCounter)
