import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
import sys
sys.path.append('..')
from human_data import get_scheurich_etal_2018_data_and_result 

# time parameters
fs   = 500
T    = 1/fs
dur  = 25
time = np.arange(0,dur,T)
halfsamps = np.floor(len(time)/2);

# oscillator parameters
a_d = -1
b_d = 4
b2_d = -0.25
F_d = 1.5
a  = 1
b  = -1
l1 = 0.75 # learning rate
l2 = 0.04 # elasticity
l2_d = l2
base = 2
z  = (1.0+0.0j)*np.ones(time.shape) # initial conditions
d = (0.0+0.0j)*np.ones(time.shape) # initial conditions

# human data (Scheurich et al. 2018)
subjs_data = get_scheurich_etal_2018_data_and_result()
musicians = subjs_data[0] # group musicians

mean_indiv = np.zeros((len(musicians), 4)) 
locs_z = pks_z = locs_F = pks_F  = []        # Locations @ local maxima
for ispr, spr in enumerate(musicians):
    print(spr) 
    # generate period lengths ms 30% faster 15% ... to a given SPR
    stim_freqs = np.linspace(0.7, 1.30, 5) * spr   # Metronome's period length in milliseconds
    stim_freqs = np.delete(stim_freqs, 2, axis=0)  # remove central element
    mean_asyn_freqs = np.zeros(stim_freqs.shape)
    
    # iterate over stimulus frequencies
    for i, freq in enumerate(stim_freqs):
        f = np.zeros(time.shape);         # Adaptive frequency (Hebbian)
        f[0] = 1000/spr;                  # Get musician's SPR
        f_d = f
        F = np.exp(1j * 2 * np.pi * time * (1000/freq))  # Stimulus "Metronome"
        locs_z = pks_z = locs_F = pks_F  = []       # Locations @ local maxima
        
        # Forward Euler integration
        for n, t in enumerate(time[:-1]):
            d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2)) + b2_d*np.power(np.abs(d[n]),4)/(1-np.power(np.abs(d[n]),2))) + F_d*F[n])
            f_d[n+1] = f_d[n] + T*(-f_d[n]*l1*np.real(F_d*F[n])*np.sin(np.angle(d[n])) - l2_d*(np.power(base,f_d[n])-np.power(base,f[n]))/base)
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + d[n])
            f[n+1] = f[n] + T*(-f[n]*l1*np.real(d[n])*np.sin(np.angle(z[n])) - l2*(np.power(base,f[n]) - np.power(base,f[0]))/base)
            
            # Find peaks a.k.a local maxima - (zero crossing)
            if (z[n+1].imag >= 0.0) and (z[n].imag <= 0.0):
                locs_z  = np.append(locs_z, n+1)
                pks_z   = np.append(pks_z, z[n+1].real)
            if (F[n+1].imag >= 0.0) and (F[n].imag <= 0.0):
                locs_F  = np.append(locs_F, n+1)
                pks_F   = np.append(pks_F, z[n+1].real)
                
        np.insert(locs_F, 0, 1)

        #plt.subplot(2,1,1)
        #plt.plot(np.real(z[:14000]))
        #plt.plot(np.real(F[:14000]))
        #plt.plot(1/f[:14000])
        #plt.grid()
        #plt.subplot(2,1,2)
        #plt.plot(np.real(d[:14000]))
        #plt.plot(np.real(F[:14000]))
        #plt.plot(1/f_d[:14000])
        #plt.grid()
        #plt.show()
        
        # which z peak is closest to the midpoint of the simulation?
        halfsamps_locsz_diff = np.absolute(halfsamps - locs_z)
        mid_nzpeak_index = np.argmin(halfsamps_locsz_diff) # get index of minimum @ half samp
        mid_nzpeak = locs_z[mid_nzpeak_index]

        # eliminate the first half of the simulation for z
        locs_z = locs_z[mid_nzpeak_index:]

        # which F peak is closest to mid_nzpeak?
        mid_nzpeak_locs_F_diff = np.absolute(locs_F - mid_nzpeak)
        mid_F_peaks_index = np.argmin(mid_nzpeak_locs_F_diff)

        # which z peak is penultimate?
        pen_nzpeak = locs_z[-2]
        # which F peak is closest to the penultimate z peak?
        pen_nzpeak_locs_F_diff = np.absolute(locs_F - pen_nzpeak)
        pen_F_peaks_index = np.argmin(pen_nzpeak_locs_F_diff)

        # compute the mean asynchrony
        mean_asynchrony = locs_z[0:-2] - locs_F[mid_F_peaks_index:pen_F_peaks_index]
        mean_asyn_freqs[i] = 1000 * mean_asynchrony.mean(0)/fs
        
    mean_indiv[ispr,:] = mean_asyn_freqs

mean_asynchronies = mean_indiv.mean(0);
plt.bar(np.arange(len(mean_asynchronies)), mean_asynchronies)
plt.show()
