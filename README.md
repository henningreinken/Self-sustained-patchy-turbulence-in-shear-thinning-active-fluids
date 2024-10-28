<h3>This repository contains Python scripts used to investigate the turbulence patterns in shear-thinning active fluids.</h3>


<h2>REQUIREMENTS</h2>

Python 3

tested with Python 3.10.12


<h2>USAGE INSTRUCTIIONS</h2>

All scripts are executed via the command line from within the main folder. For simplicity, parameter values are set directly within the scripts.

The scripts starting "*simulation...*" are used for the main calculations. They solve the evolution equation for the velocity field via a pseudo spectral method and perform additional analysis of the dynamics.

*simulation_generateInitialValues.py* <br>
This script always has to be executed first to generate the initial values for subsequent calculations. It starts with a given value of the activity parameter, determines the spatiotemporal dynamics and saves the resulting velocity field. Then, the activity parameter is decreased by a small amount and the calculation is performed again. These steps are repeated until a given value of activity is reached.

*simulation_continue.py* <br>
This script contiues the calculations for a particular activity starting from the initial values determined before via *simulation_generateInitialValues.py* . It also saves system-averaged quantities such as the mean enstrophy.

*simulation_determineVelocityAndViscosityDistributions.py* <br>
This script continues the calculations for a particular activity and determines the distributions of velocity and coarse-grained viscosity.

*simulation_determineDistributionsCriticalPoint.py* <br>
This script continues the calculations for a particular activity and determines the distributions of gap length and quiescent time. These are particularly relevant close to the critical point.

*averageData.py* <br>
This script averages the data determined via *simulation_continue.py*, e.g., the mean enstrophy and turbulence fraction.

*calculateKurtosis.py* <br>
This script calculates the kurtosis from the velocity disctribution data.

*determineDistrEllAndTau.py* <br>
This script determines the gap length and quiescent time distributions from results of multiple simulations performed via *simulation_determineDistributionsCriticalPoint.py* and formats the data in plot-friendly way.

*determineCritExp.py* <br>
This script performs a fitting procedure to obtain the critical exponents near the critical point from the gap length and quiescent time distributions.

*determineExpTurbFrac.py* <br>
This script performs a fitting procedure to obtain the exponent of the power-law scaling of the turbulence fraction with distance to the critical point.



