# -*- coding: utf-8 -*-
"""
Descripcion: Main code of the SBF_uncertainty_estimator
Interacts with user to acquire the inputs, then runs the SBF inference iteratively.
Calculates the SBF uncertainty among the iterations and displays the results on terminal.
@author: Pablo Rodríguez Beltrán
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm  
from functools import partial
from scipy.optimize import curve_fit
from utilities import *

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_number(prompt, case_flag):
    input_value = input(prompt)
    if is_number(input_value):
        if case_flag == 0:
            input_value = int(float(input_value))
        if case_flag == 1:
            input_value = float(input_value)
            
        if (input_value <= 0):
            print("ERROR: Input must be a positive and non-zero number.")
            return(get_number(prompt, case_flag))   
        else:
            return input_value
    else:
        print("ERROR: Input must be a positive and non-zero number.")
        return(get_number(prompt, case_flag))      

def get_str(prompt):
    input_value = input(prompt)
    if input_value == "n" or input_value == "y": 
        return input_value
    else:
        print("ERROR: Please use \'y\' or \'n\', not any other character.")
        return(get_str(prompt))

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

##### Assures different seeds for random functions
def init_pool():
    np.random.seed(multiprocessing.current_process().pid)
    return

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

def perform_SBF_inference(it):

    ###########################################################################################################################################
    ###################################################### CREATING THE MOCK GALAXY ###########################################################
    ###########################################################################################################################################
    
    ##### Template image to operate with
    num_cols=int(tot_pix) 
    num_rows=int(tot_pix) 
    ZeroMatrix = np.zeros((num_cols,num_rows))
    
    ##### Two-dimensional galaxy model (mean galaxy)
    galModel_mean = sersic_image(ZeroMatrix, DN_eff, n_Sersic, 0., R_eff_pix, 0., num_rows/2, num_cols/2)

    ##### Adding the fluctuation of the stellar population luminosity
    galModel_fluc = SBF_noise(galModel_mean, DN_sbf) 
    
    ##### Adding the sky background 
    galModel_fluc_sky = galModel_fluc + sky_back 
    
    ##### Creating and appliying the PSF
    PSF = gaussian_PSF(PSF_radius) 
    galModel_fluc_sky_PSF = convolvenorm_fft_wrap(galModel_fluc_sky, PSF) 
    galModel_meanPSF = convolvenorm_fft_wrap(galModel_mean, PSF) 
        
    ##### Appliying the readout noise
    gal_mock = AddNoise(galModel_fluc_sky_PSF) 

    ###### This is ends the creation of the mock galaxy

    ###########################################################################################################################################
    ############################################################### SBF DERIVATION ############################################################
    ###########################################################################################################################################
    
    ##### Extracting the fluctuation image:
    ##### Subtracting the sky background
    gal_mock_NoSky = gal_mock - sky_back 
    ##### Subtracting the mean model convolved with the PSF, this is the fluctuation image
    gal_mockFluc = gal_mock_NoSky - galModel_meanPSF   
    ##### Normalizing the fluctuation image
    gal_mockFluc_norm = gal_mockFluc / np.sqrt(galModel_meanPSF)        
    
    ##### Modelling the readout noise
    readout_noise_model = AddNoise(sky_back + galModel_meanPSF) - (sky_back + galModel_meanPSF)   
    ##### Normalizing the modelled readout noise
    readout_noise_model_norm =  readout_noise_model / np.sqrt(galModel_meanPSF) 

    ##### Creating and appliying the mask:
    ##### Creating an annular mask
    mask = RadialMask(ZeroMatrix, r1, r2, int(num_cols/2), int(num_rows/2))
    ##### Appliying the mask to the mock galaxy image
    gal_mock_mask = gal_mock * mask
    ##### Appliying the mask to the normalized fluctuatiuon image
    gal_mockFluc_norm_mask = gal_mockFluc_norm * mask
    ##### Appliying the mask to the modelled readout noise
    readout_noise_model_norm_mask = readout_noise_model_norm * mask

    ####################################################### MOVING INTO FOURIER SPACE ######################################################
    
    ##### Azimuthally averaged power spectrum of the galaxy fluctuation
    rad_PS_gal_mockFluc_norm_mask = PowerSpecArray(gal_mockFluc_norm_mask)

    ##### Azimuthally averaged power spectrum of the readout noise model, normalized and masked
    rad_PS_readout_noise_model_norm_mask = PowerSpecArray(readout_noise_model_norm_mask)   

    ##### Azimuthally averaged power spectrum of the mask
#     rad_PS_mask = PowerSpecArray(mask)/(num_cols * num_rows) # Normalized   

    ##### Procedure for the masked PSF
    ##### Power spectrum image of the mask
    PS_mask = PowerSpecData(mask)/(num_cols * num_rows)
    ##### Power spectrum image of the resized PSF
    ReSize_PSF = PSF2Image(ZeroMatrix, PSF)   
    PS_PSF = PowerSpecData(ReSize_PSF) 
    ##### Power spectrum image of the resized PSF 
    Resized_PS_maskedPSF = convolvenorm_fft(PS_PSF, PS_mask) * num_cols * num_rows
    ##### Azimuthally averaged power spectrum of the masked PSF
    rad_PS_Resized_PS_maskedPSF = radial_profile(Resized_PS_maskedPSF, [num_rows/2,num_cols/2]) 
    
    ##### Azimuthally averaged power spectrum of the PSF 
    rad_PS_ReSize_PSF = PowerSpecArray(ReSize_PSF) * num_cols * num_rows # Normalized 
      
    ####################################################### PERFORMING THE FITTING ######################################################
    
    ##### Definition of the function to fit
    def sbf_to_fit(x, sbf, final_ps_sky, rad_PS_Resized_PS_maskedPSF):
        return rad_PS_Resized_PS_maskedPSF * sbf + rad_PS_Resized_PS_maskedPSF + final_ps_sky

    ##### Feeding known parameters to the incoming fitting, leaving the SBF as sole parameter to infer
    sbf_to_fit_re = partial(sbf_to_fit, final_ps_sky=rad_PS_readout_noise_model_norm_mask[kfit_i:kfit_f])
    sbf_to_fit_reRe = partial(sbf_to_fit_re, rad_PS_Resized_PS_maskedPSF=rad_PS_Resized_PS_maskedPSF[kfit_i:kfit_f])
    ##### Performing the fitting in the range of frequencies selected
    DN_sbf_fit, cov_DN_sbf_fit = curve_fit(f=sbf_to_fit_reRe, xdata=np.linspace(kfit_i,kfit_f,len(rad_PS_gal_mockFluc_norm_mask[kfit_i:kfit_f])), ydata=rad_PS_gal_mockFluc_norm_mask[kfit_i:kfit_f])   

    ####################################################### EVALUATING THE FITTING ######################################################

    ##### Calculating the standard deviation of the fitting
    sigma_rel_sbf = 100. * np.sqrt(cov_DN_sbf_fit)/DN_sbf_fit  
    ##### Calculating the relative error
    err_rel_sbf = 100. * abs(DN_sbf-DN_sbf_fit)/DN_sbf

    ################################################ CHECKING CONDITION FOR LOW FLUX PIXELS #############################################
    
    ##### Apply the threshold and get a mask for pixels below the threshold
    below_threshold_mask = (gal_mock < 10. * DN_sbf).astype(int) * mask
    ##### Count the number of non-masked pixels
    num_pixels_mask = np.count_nonzero(gal_mock_mask)
    ##### Count the number of pixels below the threshold
    num_pixels_below_threshold = np.count_nonzero(below_threshold_mask)
    ##### Calculate ratio of pixels that do not fullfil the condition
    ratio_ill_pixels = 100. * (num_pixels_below_threshold / num_pixels_mask)
    ##### Calculate the mean of pixel values below the threshold respect to the SBF value
    if num_pixels_below_threshold == 0:
        mean_gal_mock_ill_pixels_value = 0.
        ratio_below_threshold = 0.
    else:
        mean_gal_mock_ill_pixels_value = np.sum(np.sum(gal_mock * below_threshold_mask)) / num_pixels_below_threshold
        ratio_below_threshold = 100. * mean_gal_mock_ill_pixels_value / (10. * DN_sbf)
        
    ###########################################################################################################################################
    ######################################################### CALLING PLOTTING FUNCTION #######################################################
    ###########################################################################################################################################
        
    if it == nIt-1 and display_flag == "y" and parallel_flag == "n":
        from plotting_function import plot_gal
        plot_gal(galModel_mean, gal_mock, gal_mockFluc_norm_mask, rad_PS_gal_mockFluc_norm_mask, rad_PS_readout_noise_model_norm_mask, 
                 rad_PS_Resized_PS_maskedPSF, rad_PS_ReSize_PSF, DN_sbf, DN_sbf_fit, readout_noise_model, sky_back, R_eff_pix, kfit_i, kfit_f)                    

    ###########################################################################################################################################
    ###########################################################################################################################################
    ###########################################################################################################################################

    return DN_sbf_fit, sigma_rel_sbf, err_rel_sbf, ratio_ill_pixels, ratio_below_threshold

###########################################################################################################################################
################################################################## USER INPUTS ############################################################
###########################################################################################################################################

print("Hello! This program is a simple estimator of the SBF uncertainty.")
print()
print("Do you want to display the considerations taken for the calculation?")
display_flag = get_str("(y or n) = ")
  
if display_flag == "y":
    considerations ="""
    This code considers a Sersic profile, a fluctuation of the stellar population luminosity as a random Gaussian distribution, a flat sky, 
    a Gaussian PSF and a random Poisson distribution for the readout noise. No GC nor background sources are considered. The code inputs are: 
    the number of counts at the effective radius, the number of counts associated to the SBF, the sky counts, the size of the point spread 
    function in pixels, the size of the image in pixels, the effective radius in pixels, the Sersic index and the annular mask applied in pixels too.
    We also ask for the range of frequencies where the fitting will be performed. We recommend checking a PS image output to assure proper fitting 
    ranges are selected. Finally, we ask for the number of Monte Carlo simulations to be done, where the randomness comes from the stellar 
    population luminosity fluctuation and the readout noise. 
      
    The code returns estimations of the uncertainty according to the number simulations taken: the accuracy (the 90% percentile of the relative error) 
    the fitting quality (the 90% percentile of the standard deviation of the fitting) and the precision (the relative 90% width of the distribution). 
    Note that, if you introduce a real SBF value retrieved from the inference of an actual observation, the relative error is not a measure of the accuracy 
    of the true SBF value of the galaxy, but a comparison of your inferred SBF and the Monte Carlo simulations.  
      
    We check for low luminosity pixels (Gal mock mask < 10 x SBF) where the modelling is not necessarily correct and warn the user if necessary.  
    Also, we provide the mean value of the fitted SBFs. We acknowledge the program is a very rough estimator of the SBF uncertainty, 
    still, as an ideal laboratory, it serves as low threshold for the uncertainty and  observations will probably retrieve worse results.  
    """
    print(considerations)
elif display_flag == "n":
    print()
 
print("The methodology applied is explained with detail in the associated paper:")
print("Modelling of Surface Brightness Fluctuation inference. Methodology, uncertainty and recommendations.")
print("P. Rodríguez-Beltrán, M. Cerviño, A. Vazdekis and M. A. Beasley")
print()
print("We encourage the users to adapt and modify the code in a way that accomplishes their wishes.")
print()
print("Please introduce the following inputs: ")
print("(inputs in [px] or [px^-1] units will be forced to integer)")
print()

print("Size of the image where the galaxy will be centered: ")
tot_pix = get_number("n pix [px] = ", 0)
print()

print("Effective radius of the mock galaxy in pixels: ") # ojo que debe ser menor que la mitad de npix
R_eff_pix = get_number("Reff [px] = ", 0)
if (R_eff_pix > tot_pix / 2):
    print("WARNING! Reff greater than image size. Program might break.")
print()

print("Size of the PSF, as three times the standard deviation of a Gaussian: ")
PSF_radius = get_number("3 x sigma_PSF [px] = ", 0)
print()

print("Sersic profile index:")
n_Sersic = get_number("n = ", 1)
b_n = 2. * n_Sersic - 0.324   # for n > 1
print()

print("Number of counts in the effective radius of the galaxy :")
DN_eff = get_number("N_Reff [counts] = ", 1)
print()

print("Number of counts associated to the SBF:")
DN_sbf = get_number("N_SBF [counts] = ", 1)
print()

print("Number of counts of the sky background:")
sky_back = get_number("N_Sky [counts] = ", 1)
print()

print("The mask considered is annular, determined by an inner and an outer radius (r1,r2) in pixels.")
r1 = get_number("r1 [px]: ", 0)
if (r1 > tot_pix / 2):
    print("WARNING! \'r1\' greater than image size. Program might break.")
r2 = get_number("r2 [px]: ", 0)
if (r2 > tot_pix / 2):
    print("WARNING! \'r2\' greater than image size. Program might break.")
if (r1 > r2):
    print("WARNING! \'r1\' greater than \'r2\'. Program might break.") 
print()

print("Range of frequencies (kfit_i,kfit_f ) where the fitting will be performed.")
print("We recommend checking a PS image output to assure proper fitting ranges are selected.")
kfit_i = get_number("kfit_i [px^-1] = ", 0)
if (kfit_i > tot_pix / 2):
    print("WARNING! \'kfit_i\' greater than image size. Program might break.")
kfit_f = get_number("kfit_f [px^-1] = ", 0)
if (kfit_f > tot_pix / 2):
    print("WARNING! \'kfit_f\' greater than image size. Program might break.")
if (kfit_i > kfit_f):
    print("WARNING! \'kfit_i\' greater than \'kfit_f\'. Program might break.") 
print()

print("Number of Monte Carlo iterations: ")
nIt = get_number("N_it = ", 0)
print()

print("Display radial power spectrum image of the final inference: ")
display_flag = get_str("(y or n) = ")
print()

print("Run in parallel (parallel option does not display the images): ")
parallel_flag = get_str("(y or n) = ")
print()

print("Calculating... (depending on the size it might take some time)")

###########################################################################################################################################
######################################################### RUNNING THE SBF UNCERTAINTY ESTIMATOR ###########################################
###########################################################################################################################################

DN_sbf_fit_list = np.zeros((nIt))
sigma_rel_list = np.zeros((nIt))
err_rel_list = np.zeros((nIt))
ratio_ill_pixels_list = np.zeros((nIt))
ratio_below_threshold_list = np.zeros((nIt))

t0_total = time.time()

if parallel_flag == "n":
    
    for it in range(nIt):
        print("======================")
        print("Iteration = ",it)
        t0 = time.time()
        ##### Running
        DN_sbf_fit, sigma_rel_sbf, err_rel_sbf, ratio_ill_pixels, ratio_below_threshold = perform_SBF_inference(it)  
        
        ##### Saving the results of each iteration    
        DN_sbf_fit_list[it] = DN_sbf_fit
        sigma_rel_list[it] = sigma_rel_sbf
        err_rel_list[it] = err_rel_sbf
        ratio_ill_pixels_list[it] = ratio_ill_pixels
        ratio_below_threshold_list[it] = ratio_below_threshold

        print("Consumed time of this iteration [s] = " ,time.time()-t0)

###########################################################################################################################################

elif parallel_flag == "y":
    ncpus = multiprocessing.cpu_count()
    ##### Running
    if __name__ == '__main__':
        with Pool(ncpus, initializer=init_pool) as p:
            results = np.array([result_i for result_i in tqdm(p.imap_unordered(perform_SBF_inference, range(0, nIt, 1)), total=nIt)])
            
    ##### Saving the results of each iteration    
    for it in range(nIt):    
        DN_sbf_fit_list[it] = results[it][0]
        sigma_rel_list[it] = results[it][1]
        err_rel_list[it] = results[it][2]
        ratio_ill_pixels_list[it] = results[it][3]
        ratio_below_threshold_list[it] = results[it][4]
                
print("======================")

###########################################################################################################################################
########################################################## CALCULATE MEANS & PERCENTILES ##################################################
###########################################################################################################################################

DN_sbf_fit_mean = np.mean(DN_sbf_fit_list)
sigma_rel_90 = np.percentile(sigma_rel_list, 90)
err_rel_90 = np.percentile(err_rel_list, 90)
ratio_ill_pixels_90 = np.percentile(ratio_ill_pixels_list, 90)
ratio_below_threshold_90 = np.percentile(ratio_below_threshold_list, 90)

sorted_DNsbf_fit = np.sort(DN_sbf_fit_list)
min5_DNsbf_fit = sorted_DNsbf_fit[np.int(nIt*0.05)]
max95_DNsbf_fit = sorted_DNsbf_fit[np.int(nIt*0.95)]
asymErr_DN = [[DN_sbf_fit_mean-min5_DNsbf_fit], [max95_DNsbf_fit-DN_sbf_fit_mean]]
Delta_90 = 100. * (max95_DNsbf_fit-min5_DNsbf_fit) / np.mean(DN_sbf_fit_mean)

###########################################################################################################################################
############################################################## DISPLAY RESULTS ############################################################
###########################################################################################################################################

print("SBF uncertainty estimation completed!")
print()
print("Total time [s] = " ,time.time()-t0_total)
print()
print("Results obtained from the distribution of iterated inferences.")
print()
print("Mean value of the fitted SBF [counts] = ", DN_sbf_fit_mean)
print("Inferior and superior errors of the mean value of the fitted SBF [counts] = ", asymErr_DN)
print()
print("Precision of the inference. 90% width of the fitted SBF distribution [%] = ", Delta_90)
print()
print("Quality of the fitting. 90% percentile of the relative standard deviation of the fitting [%] = ", sigma_rel_90)
print()
print("Accuracy of the fitting. 90% percentile of the relative error [%] = ", err_rel_90)
print()
if ratio_ill_pixels_90 == 0:
    print("Condition fullfilled: (Gal mock mask > 10 x SBF). There is enough counts in every pixel.")
elif ratio_ill_pixels_90 > 0:
    print("WARNING! There is NOT enough counts in every pixel. Condition NOT fullfiled: (Gal mock mask < 10 x SBF).")
    print("Your inference might be biased. The modelling is not necessarily physically correct.")
    print("90% percentile of the ratio of ill pixels respect to the total non-masked pixels [%] = ",ratio_ill_pixels_90)
    print("90% percentile of the ratio of the mean of pixel values below the threshold respect to the SBF value [%] = ",ratio_below_threshold_90)
else:
    print("Exception error. Negative number.")
        
plt.show()

