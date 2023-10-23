################################################################################
# SBF_uncertainty_estimator
################################################################################

This program is a simple estimator of the SBF uncertainty based in a set of simulations of similar galaxies. It creates a Monte Carlo set of mock galaxies where a known fluctuation is introduced. Then, such fluctuation is fitted through the SBF inference methodology. At last, outcomes from the distribution of results are displayed.

We acknowledge the program is a very rough estimator of the SBF uncertainty, 
still, as an ideal laboratory, it serves as low threshold for the uncertainty, 
then, observations will probably retrieve worse results.  

The methodology applied is explained with detail in the associated A&A paper:
Modelling of Surface Brightness Fluctuation inference. Methodology, uncertainty
and recommendations.
Authors: P. Rodríguez-Beltrán, M. Cerviño, A. Vazdekis and M. A. Beasley

We encourage the users to adapt and modify the code in a way 
that accomplishes their wishes!

################################################################################
## Installation
################################################################################

For the moment, no installation is necessary. Just download the code and run it.
External packages ARE necessary:
time, numpy, matplotlib.pyplot, multiprocessing, multiprocessing.Pool, tqdm.tqdm,   
functools.partial, scipy.optimize.curve_fit, scipy.signal.fftconvolve, astropy.convolution.convolve_fft, astropy.modeling.models.Sersic2D, astropy.modeling.models.Gaussian2D, matplotlib.colors, matplotlib.patheffects

################################################################################
## Usage
################################################################################

'''
python3 SBF_uncertainty_estimator_MAIN.py 
'''
To initiate the interaction run the main code 'SBF_uncertainty_estimator_MAIN.py' in a console. Introduce the inputs that the code requires, wait for the calculation and retrieve the outputs through console.

################################################################################
## Contributing
################################################################################

Pablo R. Beltrán
Miguel Cerviño Saavedra

################################################################################
## License
################################################################################

Use freely and adapt freely to your interests.

################################################################################