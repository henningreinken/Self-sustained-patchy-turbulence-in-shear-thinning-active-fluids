
import os
import numpy as np
import sys
import time
from scipy import interpolate

# get current directory
thisPath = os.getcwd()

# coefficients
act = 1.42
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
# tEnd = 1500.0
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

dtAna = 10.0
dnAna = round(dtAna/dt)
sizeAna = round(tEnd/dtAna)
nAna = 0

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

# arrays to save data
dataAnaTime = np.zeros((sizeAna,7))

# to estimate remaining time
timeSinceLastPrint = time.time()

# parameters and quantities for coarse-graining
NCG = int(sizeInWavelength)
dPointsCG = int(N/NCG)
nExtend = dPointsCG + 1
viscExtended = np.zeros((N+2*nExtend,N+2*nExtend))
vAbsExtended = np.zeros((N+2*nExtend,N+2*nExtend))
enstrExtended = np.zeros((N+2*nExtend,N+2*nExtend))

viscCG = np.zeros((N,N))
shearRateCG = np.zeros((N,N))

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
		# print('area fraction turbulence = ' + str(1.0 - (np.mean(visc) - 1.0)/(visc0 - 1.0)), flush=True)
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
		visc = 1.0 + (visc0 - 1.0)/(1.0 + np.power(shearRate/shearRateRef,exponent))
		shearRate = np.power(2.0*vx_x*vx_x + (vx_y + vy_x)*(vx_y + vy_x) + 2.0*vy_y*vy_y,0.5)
		
		viscExtended = calcExtendedField(visc)
		
		for i in range(0,N):
			for j in range(0,N):
				iStart = round(nExtend + i - dPointsCG/2)
				iEnd = round(nExtend + i + dPointsCG/2)
				jStart = round(nExtend + j - dPointsCG/2)
				jEnd = round(nExtend + j + dPointsCG/2)
				viscCG[i,j] = np.mean(viscExtended[iStart:iEnd,jStart:jEnd])
				
		areaFracTurbViscCG = 0.0
		
		for i in range(0,N):
			for j in range(0,N):
				if viscCG[i,j] < act:
					areaFracTurbViscCG = areaFracTurbViscCG + 1.0
		areaFracTurbViscCG = areaFracTurbViscCG/float(N)/float(N)
		
		print('area fraction turbulence = ' + str(areaFracTurbViscCG), flush=True)
		print()
		
		dataAnaTime[nAna,0] = t
		dataAnaTime[nAna,1] = np.mean(np.multiply(vVort,vVort))
		dataAnaTime[nAna,2] = np.mean(np.multiply(vx,vx) + np.multiply(vy,vy))
		dataAnaTime[nAna,3] = np.mean(visc)
		dataAnaTime[nAna,4] = np.mean(shearRate)
		dataAnaTime[nAna,5] = areaFracTurbViscCG
		
		nAna = nAna + 1
		
	
# save Eulerian mean values
np.savetxt(folderName+'/dataMean%04d.dat'%(runNum),dataAnaTime)

# save last time step for the field for continuation of calculation
np.savetxt(folderName+'/datavxInit%04d.dat'%(runNum+1),vx)
np.savetxt(folderName+'/datavyInit%04d.dat'%(runNum+1),vy)