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
dur  = 50
time = np.arange(0,dur,T)
halfsamps = np.floor(len(time)/2);

# oscillator parameters
a_d = 1
b_d = -1
b2_d = 0
F_d = 1
a  = 1
b  = -1
l1 = 5 # learning rate
l2 = 1.4 # elasticity
l2_d = l2/50
base = np.exp(1)
z  = (1.0+0.0j)*np.ones(time.shape) # initial conditions
d = (0.99+0.0j)*np.ones(time.shape) # initial conditions

# human data (Scheurich et al. 2018)
subjs_data = get_scheurich_etal_2018_data_and_result()
musicians = subjs_data[0] # group musicians
#musicians = [420]

mean_indiv = np.zeros((len(musicians), 6)) 
locs_z = pks_z = locs_F = pks_F  = []        # Locations @ local maxima
for ispr, spr in enumerate(musicians):
    print(spr) 
    # generate period lengths ms 30% faster 15% ... to a given SPR
    stim_freqs = [freq*spr for freq in [1, 0.55, 0.7, 0.85, 1.15, 1.3, 1.45]]   # Metronome's period length in milliseconds
    print(stim_freqs)
    mean_asyn_freqs = np.zeros(len(stim_freqs)-1)
    
    # iterate over stimulus frequencies
    spr_ma = 0
    for i, freq in enumerate(stim_freqs):
        f = (1000/spr)*np.ones(time.shape)
        f_d = (1000/spr)*np.ones(time.shape)
        F = np.exp(1j * 2 * np.pi * time * (1000/freq))  # Stimulus "Metronome"
        locs_z = pks_z = locs_F = pks_F  = []       # Locations @ local maxima
        
        # Forward Euler integration
        for n, t in enumerate(time[:-1]):
            d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2)) + b2_d*np.power(np.abs(d[n]),4)/(1-np.power(np.abs(d[n]),2))) + F_d*F[n])
            f_d[n+1] = f_d[n] + T*f_d[n]*(-l1*np.real(F_d*F[n])*np.sin(np.angle(d[n])) - l2_d*(f_d[n]-f[n])/f[n])
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + np.exp(1j*np.angle(d[n])))
            f[n+1] = f[n] + T*f[n]*(-l1*np.cos(np.angle(d[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f[n]-f[0])/f[0])-1))
            
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
        if i == 0:
            spr_ma = 1000 * mean_asynchrony.mean(0)/fs
            print(spr_ma)
        else:
            mean_asyn_freqs[i-1] = (1000 * mean_asynchrony.mean(0)/fs) - spr_ma
        
    mean_indiv[ispr,:] = mean_asyn_freqs

print(mean_indiv)
mean_asynchronies = mean_indiv.mean(0)
error_asynchronies = mean_indiv.std(0)/np.sqrt(mean_indiv.shape[0])

fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5))
ax1.bar(np.arange(4),[10.5, 7.5, -3.5, -6.5],color='grey',edgecolor='black')
ax1.errorbar(np.arange(4),[10.5, 7.5, -3.5, -6.5],[1, 1, 1.5, 2],[0,0,0,0],'none',ecolor='black')
ax1.set_ylim([-25, 25])
ax2.set_xlim([-0.5, 3.5])
ax1.set_ylabel('Mean adjusted asynchrony (ms)',fontsize=15)
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(['F30','F15','S15','S30'],fontsize=15)
ax1.grid(color='gray', linestyle='dashed')
ax1.set_axisbelow(True)
ax1.tick_params(axis="y", labelsize=15)
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=25, weight='bold')
ax2.axvspan(-0.5,0.5, facecolor='#c0c2c4',alpha=0.1,zorder=0)
ax2.axvspan(4.5,5.5, facecolor='#c0c2c4',alpha=0.1,zorder=0)
ax2.bar(np.arange(len(mean_asynchronies)), mean_asynchronies,color='grey',edgecolor='black')
ax2.errorbar(np.arange(len(mean_asynchronies)), mean_asynchronies, error_asynchronies,[0,0,0,0,0,0],'none',ecolor='black')
ax2.set_ylim([-25, 25])
ax2.set_xlim([-0.5, 5.5])
ax2.set_axisbelow(True)
ax2.grid(color='gray', linestyle='dashed')
ax2.set_xticks(np.arange(6))
ax2.set_xticklabels(['F45','F30','F15','S15','S30','S45'],fontsize=15)
ax2.text(-0.1, 1.05, 'B', transform=ax1.transAxes, size=25, weight='bold')
plt.savefig('../figures_raw/fig2.eps')
