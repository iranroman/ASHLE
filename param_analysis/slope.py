import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import find_peaks
from scipy.stats import linregress


# time parameters
fs = 1000
T = 1/fs
dur = 50
time = np.arange(0,dur,T)

# oscillator parameters
a_m = 1
b_1m = -1
b_2m = 0
a_b = 1
b_1b = -1
f_0 = 2.5
f_stim = [1000/(x*(1000/f_0)) for x in [0.55, 0.70, 0.85, 1.15, 1.3, 1.45]]
l1s = [4] #[0.3, 0.4, 0.5, 0.6]
l2s = [2] #[0.0, 0.025, 0.05] 
all_gammas = [0.01, 0.02, 0.04, 0.08]
base = [np.exp(1)]
gsigm = 0
base_div = 1
nlearn = 0

numplots = len(l1s)*len(l2s)
plt.figure(figsize=(6,5))
gs = gridspec.GridSpec(len(l1s), len(l2s), wspace=0, hspace=0)

for gamma in all_gammas:
    for il1, l1 in enumerate(l1s):
        for il2, l2 in enumerate(l2s):
    
            bar_results = np.zeros((len(f_stim)))
            mean_slope = 0
            for if_s, f_s in enumerate(f_stim):
    
                x = np.zeros(time.shape)
                z_m = (0.01+0.0j)*np.ones(time.shape) # initial conditions
                f_m = f_s*np.ones(time.shape) # initial conditions
                z_b = (0.01+0.0j)*np.ones(time.shape) # initial conditions
                f_b = f_s*np.ones(time.shape) # initial conditions
                for n, t in enumerate(time[:-1]):

                    z_m[n+1] = z_m[n] + T*f_m[n]*(z_m[n]*(a_m + 1j*2*np.pi + b_1m*np.power(np.abs(z_m[n]),2)) + x[n])
                    f_m[n+1] = f_m[n] + T*f_m[n]*(-l1*np.real(1j*x[n]*np.conj(z_m[n])/np.abs(z_m[n])) - gamma*(np.power(base,(f_m[n]-f_b[n])/f_b[n])-1))
                    z_b[n+1] = z_b[n] + T*f_b[n]*(z_b[n]*(a_m + 1j*2*np.pi + b_1b*np.power(np.abs(z_b[n]),2)) + np.exp(1j*np.angle(z_m[n])))
                    f_b[n+1] = f_b[n] + T*f_b[n]*(-l1*np.real(1j*np.exp(1j*np.angle(z_m[n]))*np.conj(z_b[n])/np.abs(z_b[n])) - l2*(np.power(base,(f_b[n]-f_0)/f_0)-1))

                peaks, _ = find_peaks(np.real(z_b))
                peaks = 1000*peaks/fs # converting to miliseconds
                peaks = peaks[:128]
                slope, _, _, _, _ = linregress(range(len(np.diff(peaks))), np.diff(peaks))
                bar_results[if_s] = slope - mean_slope

                print(np.diff(peaks))
                print(slope)

            print(l1,l2)
            print(bar_results)
            ax = plt.subplot(gs[il1, il2])
            ax.plot(range(len(f_stim)), bar_results,'-o',linewidth=3,markersize=6,label=r'$\gamma = {}$'.format(gamma))

ax.legend(loc='lower left', prop={'size':11})
ax.grid(linestyle='dashed')
ax.set_axisbelow(True)
ax.set_ylabel('Slope', fontsize=12)
ax.tick_params(axis="y", labelsize=10)
ax.set_xticks(range(len(f_stim)))
ax.set_xticklabels(['F45','F30','F15','S15','S30','S45'],fontsize=10)
plt.savefig('../figures_raw/fig7.eps')
