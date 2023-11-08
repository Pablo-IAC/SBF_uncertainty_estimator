# -*- coding: utf-8 -*-
"""
Descripcion: Function for plotting two figures:
The resulting mock galaxy, its profile.
The azimuthally averaged power spectrum of the different components of the fitting.
@author: Pablo Rodríguez Beltrán
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from utilities import radial_profile

plt.rc("lines", linewidth=1.2)   #1.5
plt.rcParams.update({"font.size": 13})


def plot_gal(galModel_mean, gal_mock, gal_mockFluc_norm_mask, rad_PS_gal_mockFluc_norm_mask, rad_PS_readout_noise_model_norm_mask, 
             rad_PS_Resized_PS_maskedPSF, rad_PS_ReSize_PSF, DN_sbf, DN_sbf_fit, readout_noise_model, sky_back, R_eff_pix, kfit_i, kfit_f):
    
    print("Plotting")
      
    row_len, col_len = np.shape(galModel_mean)
    
    ################################################################################################################################
    #### First figure
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.3, hspace=0.1)    
    
    axA = axs[0]
    axA.set_xlabel("$X$ [px]")
    axA.set_ylabel("$Y$ [px]")
    axA.set_title("Gal$_{\mathrm{mock}}$")   
    axA.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "both")
    axA.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "both")                         
    axA.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "minor")
    axA.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "minor")  
    axA.minorticks_on()
    map = axA.pcolormesh(gal_mock, norm=colors.LogNorm(vmin=1e4, vmax=5e5)) 
    cbar = fig.colorbar(map, ax=axA)
    cbar.ax.set_ylabel("$N(X,Y)$ [Counts]")
    
    theta = np.linspace(0, 2*np.pi, 100)
    r = R_eff_pix
    x1 = r*np.cos(theta) + row_len/2
    x2 = r*np.sin(theta) + col_len/2
    axA.plot(x1, x2, color = "r", linewidth = 0.9)
    
    axB = axs[1]
    axB.set_xlabel("$r$ [px]")
    axB.set_ylabel("$N(r)$ [counts]")
    axB.set_title("Radial profiles")   
    axB.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "both", direction = "in")
    axB.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "both", direction = "in")                         
    axB.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "minor", direction = "in")
    axB.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "minor", direction = "in")  
    axB.minorticks_on()
    axB.axvline(x = R_eff_pix, color = "r")
    axB.loglog(radial_profile(galModel_mean, [row_len/2, col_len/2]), label = "Gal$_{\mathrm{mean}}$", linewidth = 2)
    axB.loglog(radial_profile(gal_mock, [row_len/2, col_len/2]), label = "Gal$_{\mathrm{mock}}$", linestyle = "dotted", linewidth = 3)
#     axB.loglog(radial_profile(readout_noise_model, [row_len/2, col_len/2])+sky_back, label = "Sky+$R_{\mathrm{aprox}}$", linestyle = "solid", linewidth = 1.5)
    axB.set_xlim([1, row_len/2])
    axB.legend(loc="best", fontsize=11)    
    
    ################################################################################################################################
    #### Second figure

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7.5))  
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.94, wspace=None, hspace=None)
          
    axs.set_xlabel("$k$ [px$^{-1}$]")
    axs.set_ylabel("Rescaled. Log$_{10}$($PS_{\mathrm{r}}$)")  
    axs.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "both", direction = "in")
    axs.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "both", direction = "in")                         
    axs.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "minor", direction = "in")
    axs.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "minor", direction = "in")  
    axs.minorticks_on()
    axs.vlines(x = kfit_i, ymin = -10, ymax = 10, linestyle = "dotted", color = "y")
    axs.vlines(x = kfit_f, ymin = -10, ymax = 10, linestyle = "dotted", color = "y")
    axs.axvspan(kfit_i, kfit_f, ymin = -10, ymax = 10, alpha=0.15, color="y")
    ratioPSFloss_k0 = rad_PS_ReSize_PSF[0]/rad_PS_Resized_PS_maskedPSF[0]
    axs.plot(np.log10(rad_PS_readout_noise_model_norm_mask)+np.log10(ratioPSFloss_k0), color = "c", linewidth = 2, label="$PS$($R_{\mathrm{aprox}}^{\mathrm{norm}}$$\cdot$Mask)$_{\mathrm{r}}$") # "PS$_{\mathrm{r}}$(R$_{aprox}$$\cdot$Mask)/$\sqrt{Gal_{mean}\otimes PSF}$"
    axs.plot(np.log10(rad_PS_gal_mockFluc_norm_mask) + np.log10(ratioPSFloss_k0), color = "b", linewidth = 2, label="$PS$(Gal$_{\mathrm{mock\;fluc\;mask}}$)$_{\mathrm{r}}$")
    axs.plot(np.log10(rad_PS_Resized_PS_maskedPSF * (DN_sbf) + rad_PS_readout_noise_model_norm_mask)+np.log10(ratioPSFloss_k0), color = "r", linewidth = 3.25, label="$\\bar{N}_{real}$ $\cdot$($PS$(PSF)$\\otimes PS$(Mask))$_{\mathrm{r}}$ + $PS$($R_{\mathrm{aprox}}^{\mathrm{norm}}$ $\cdot$Mask)$_{\mathrm{r}}$")
    axs.plot(np.log10(rad_PS_Resized_PS_maskedPSF * (DN_sbf_fit) + rad_PS_readout_noise_model_norm_mask)+np.log10(ratioPSFloss_k0), color = "g", linestyle = "--", linewidth = 1.25, 
             label="$\\bar{N}_{fit}$ $\cdot$($PS$(PSF)$\\otimes PS$(Mask))$_{\mathrm{r}}$ + $PS$($R_{\mathrm{aprox}}^{\mathrm{norm}}$ $\cdot$Mask)$_{\mathrm{r}}$", path_effects=[pe.Stroke(linewidth=2.5, foreground="k"), pe.Normal()])
    axs.legend(loc="upper right", fontsize = 11)    
    axs.set_xlim([0., row_len/2])
    axs.set_ylim([np.mean(np.log10(rad_PS_readout_noise_model_norm_mask)+np.log10(ratioPSFloss_k0))*0.9, (np.log10(rad_PS_Resized_PS_maskedPSF * (DN_sbf) + rad_PS_readout_noise_model_norm_mask)+np.log10(ratioPSFloss_k0))[0]*1.1])
  
    ################################################
    
    inner_ax = axs.inset_axes([0.6, 0.4, 0.27, 0.32])
    
    inner_ax.set_xlabel("X [px]", fontsize = 10)
    inner_ax.set_ylabel("Y [px]", fontsize = 10)
    inner_ax.set_title("Gal$_{\mathrm{mock\;fluc\;mask}}$", fontsize = 12)   
    inner_ax.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "both", direction = "in")
    inner_ax.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "both", direction = "in")                         
    inner_ax.tick_params(axis="y",left="True", labelleft="True", right="True", labelright="", which = "minor", direction = "in")
    inner_ax.tick_params(axis="x",bottom="True", labelbottom="True", top="True", labeltop="", which = "minor", direction = "in")  
    inner_ax.minorticks_on()
    gal_mockFluc_norm_mask[gal_mockFluc_norm_mask == 0] = np.nan
    map = inner_ax.pcolormesh(gal_mockFluc_norm_mask)   
    cbar = fig.colorbar(map, ax=inner_ax, cax=fig.add_axes([0.87, 0.44, 0.015, 0.22]))
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("[Counts]", fontsize = 10)
        
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    
    return 
