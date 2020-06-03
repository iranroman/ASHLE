import sys
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import find_peaks
from scipy.stats import linregress


# time parameters
fs = 500
T = 1/fs
dur = 50
time = np.arange(0,dur,T)

# oscillator parameters
a_m = -0.5
b_1m = 2.5
b_2m = -2.13
a_b = 1
b_1b = -1
f_0 = 1
f_stim = [1.5, 0.5] 
l1s = [1, 2, 4, 16, 32, 64] 
l2s = [1, 2, 4, 16, 32, 64] 
base= np.exp(1)
nlearn = 4

numplots = len(l1s)*len(l2s)
plt.figure(figsize=(10,10))
color_cmap = cm.get_cmap()
color_cmap.set_bad(color='black')

for if_s, f_s in enumerate(f_stim):
    asynch_mat = np.empty((len(l1s),len(l2s)))
    asynch_mat[:] = np.nan
    for il1, l1 in enumerate(l1s):
        for il2, l2 in enumerate(l2s):
    
            bar_results = np.zeros((len(f_stim)))
    
            x = np.exp(1j*2*np.pi*time*f_s)
            z_m = (0+0.0j)*np.ones(time.shape) # initial conditions
            f_m = np.ones(time.shape) # initial conditions
            z_b = (0+0.0j)*np.ones(time.shape) # initial conditions
            f_b = np.ones(time.shape) # initial conditions
            for n, t in enumerate(time[:-1]):
                
                z_m[n+1] = z_m[n] + T*f_m[n]*(z_m[n]*(a_m + 1j*2*np.pi + b_1m*np.power(np.abs(z_m[n]),2) + b_2m*np.power(np.abs(z_m[n]),4)/(1 - np.power(np.abs(z_m[n]),2))) + x[n])
                f_m[n+1] = f_m[n] + T*f_m[n]*(-l1*np.real(x[n])*np.sin(np.angle(z_m[n])) - l2*(np.power(base,f_m[n]) - np.power(base,f_b[n]))/np.power(base,f_b[n]))
                z_b[n+1] = z_b[n] + T*f_b[n]*(z_b[n]*(a_m + 1j*2*np.pi + b_1b*np.power(np.abs(z_b[n]),2)) + np.exp(1j*np.angle(z_m[n])))
                f_b[n+1] = f_b[n] + T*f_b[n]*(-l1*np.cos(np.angle(z_m[n]))*np.sin(np.angle(z_b[n])) - l2*(np.power(base,f_b[n]) - np.power(base,f_0))/np.power(base,f_0))
            
            locs_z, _ = find_peaks(np.real(z_b))
            locs_x, _ = find_peaks(np.real(x))
    
            # which z peak is closest to the midpoint of the simulation?
            halfsamps_locsz_diff = np.absolute(int(nlearn*fs/f_s) - locs_z)
            mid_nzpeak_index = np.argmin(halfsamps_locsz_diff) # get index of minimum @ half samp
            mid_nzpeak = locs_z[mid_nzpeak_index]
    
            # eliminate the first half of the simulation for z
            locs_z = locs_z[mid_nzpeak_index:]
    
            # which x peak is closest to mid_nzpeak?
            mid_nzpeak_locs_F_diff = np.absolute(locs_x - mid_nzpeak)
            mid_F_peaks_index = np.argmin(mid_nzpeak_locs_F_diff)
    
            # which z peak is penultimate?
            pen_nzpeak = locs_z[-2]
            # which x peak is closest to the penultimate z peak?
            pen_nzpeak_locs_F_diff = np.absolute(locs_x - pen_nzpeak)
            pen_F_peaks_index = np.argmin(pen_nzpeak_locs_F_diff)
    
            # compute the mean asynchrony
            peaks_diff = len(locs_z[0:-2]) - len(locs_x[mid_F_peaks_index:pen_F_peaks_index])
            
            if peaks_diff == 0:
                asynch_mat[il1,il2] = 1000*np.mean((locs_z[0:-2] - locs_x[mid_F_peaks_index:pen_F_peaks_index])/fs) 
    
    plt.subplot(1,2,if_s+1)
    plt.imshow(asynch_mat, vmin=-500,vmax=500)
    for il1, l1, in enumerate(l1s):
        for il2, l2, in enumerate(l2s):
            if not np.isnan(asynch_mat[il1,il2]):
                plt.text(il2, il1, str(int(asynch_mat[il1,il2]))+'ms', ha='center', va='center', weight='bold')
    plt.yticks(range(len(l1s)),l1s)
    plt.xticks(range(len(l2s)),l2s)
    ax = plt.gca()
    plt.text(-0.1, 1.1, string.ascii_uppercase[if_s], size=20, transform=ax.transAxes, weight='bold')

plt.show()
