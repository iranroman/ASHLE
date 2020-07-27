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
F_d = 0.1
a  = 1
b  = -100
l1 = 4 # learning rate
l2 = 2 # elasticity
l2_d = l2/50
base = np.exp(1)
gsigm = 0
sigma = 0.0006
z  = (0.0+0.0j)*np.ones(time.shape) # initial conditions
d = (0.0+0.0j)*np.ones(time.shape) # initial conditions

# human data (Scheurich et al. 2018)
subjs_data = get_scheurich_etal_2018_data_and_result()
musicians = subjs_data[0] # group musicians
#[musicians = [420]

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
            
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*F[n] + np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1))
            f[n+1] = f[n] + T*f[n]*(-l1*(np.real(F_d*F[n])*np.sin(np.angle(z[n])) - np.imag(F_d*F[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f[n]-f[0])/f[0])-1))

            #d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2))) + F_d*F[n])
            #f_d[n+1] = f_d[n] + T*f_d[n]*(-l1*np.real(F_d*F[n])*np.sin(np.angle(d[n])) - (l1/((f[0])*600))*np.cos(np.angle(z[n]))*np.sin(np.angle(d[n])))
            #z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + np.exp(1j*np.angle(d[n])))
            #f[n+1] = f[n] + T*f[n]*(-l1*np.cos(np.angle(d[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f[n]-f[0])/f[0])-1))
            
        # Find peaks a.k.a local maxima - (zero crossing)
        locs_z, _ = find_peaks(np.real(z), prominence=0.1)
        locs_F, _ = find_peaks(np.real(F))
                
        np.insert(locs_F, 0, 1)

        #plt.subplot(2,1,1)
        plt.plot(np.real(z[:14000]))
        plt.plot(np.real(F_d*F[:14000]))
        plt.plot(1/f[:14000])
        plt.grid()
        #plt.subplot(2,1,2)
        #plt.plot(np.real(d[:14000]))
        #plt.plot(np.real(F[:14000]))
        #plt.plot(1/f_d[:14000])
        #plt.grid()
        plt.show()
        
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

mean_asynchronies = mean_indiv.mean(0)
error_asynchronies = mean_indiv.std(0)/np.sqrt(mean_indiv.shape[0])

fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True,figsize=(10,5))
ax1.bar(np.arange(4),[10.5, 7.5, -3.5, -6.5],color='grey',edgecolor='black')
ax1.errorbar(np.arange(4),[10.5, 7.5, -3.5, -6.5],[1, 1, 1.5, 2],[0,0,0,0],'none',ecolor='black')
ax1.set_ylim([-25, 25])
ax2.set_xlim([-0.5, 3.5])
ax1.set_ylabel('Mean adjusted asynchrony (ms)',fontsize=15)
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(['F30','F15','S15','S30'],fontsize=12)
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
ax2.set_xticklabels(['F45','F30','F15','S15','S30','S45'],fontsize=12)
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=25, weight='bold')

all_sprs = np.linspace(350,650,5)
mean_indiv = np.zeros((len(all_sprs), 6)) 
for ispr, spr in enumerate(all_sprs):
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
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*F[n] + np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1))
            f[n+1] = f[n] + T*f[n]*(-l1*(np.real(F_d*F[n])*np.sin(np.angle(z[n])) - np.imag(F_d*F[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f[n]-f[0])/f[0])-1))
            #z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*F[n] + np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1))
            #f[n+1] = f[n] + T*f[n]*(-l1*np.real(F_d*F[n])*np.sin(np.angle(z[n])) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f[n]-f[0])/f[0])-1))
            #d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2))) + F_d*F[n])
            #f_d[n+1] = f_d[n] + T*f_d[n]*(-l1*np.real(F_d*F[n])*np.sin(np.angle(d[n])) - (l1/((f[0])*600))*np.cos(np.angle(z[n]))*np.sin(np.angle(d[n])))
            #z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + np.exp(1j*np.angle(d[n])))
            #f[n+1] = f[n] + T*f[n]*(-l1*np.cos(np.angle(d[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f[n]-f[0])/f[0])-1))
            
        # Find peaks a.k.a local maxima - (zero crossing)
        locs_z, _ = find_peaks(np.real(z), prominence=0.1)
        locs_F, _ = find_peaks(np.real(F))
                
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

allslopes = mean_indiv
for i in range(len(all_sprs)):
    ax3.plot(range(6),allslopes[i],'-o',linewidth=3,markersize=6,label=str(int(all_sprs[i]))+'ms')
ax3.legend(loc='lower left',prop={'size': 11})
ax3.set_ylim([-25, 35])
ax3.set_xlim([-0.5, 5.5])
ax3.set_axisbelow(True)
ax3.grid(color='gray', linestyle='dashed')
ax3.set_xticks(np.arange(6))
ax3.tick_params(axis="y", labelsize=15)
ax3.set_xticklabels(['F45','F30','F15','S15','S30','S45'],fontsize=12)
ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, size=25, weight='bold')
plt.savefig('../figures_raw/fig2.eps')
