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
dur = 100
time = np.arange(0,dur,T)

# oscillator parameters
a_m = 1
b_1m = -1
b_2m = 0
a_b = 1
b_1b = -1
f_0 = 2.27
f_stim = [x*f_0 for x in [1, 1.71930111, 1.33295354, 0.76859625, 0.55374987]]
l1s = [4] #[0.3, 0.4, 0.5, 0.6]
l2s = [0.8] #[0.0, 0.025, 0.05] 
bases = [np.exp(1)]
base_div = 1
nlearn = 0

numplots = len(l1s)*len(l2s)
plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(len(l1s), len(l2s), wspace=0, hspace=0)

for base in bases:
    for il1, l1 in enumerate(l1s):
        for il2, l2 in enumerate(l2s):
    
            bar_results = np.zeros((len(f_stim)-1))
            mean_slope = 0
            for if_s, f_s in enumerate(f_stim):
    
                x = np.zeros(time.shape)
                z_m = (0.99+0.0j)*np.ones(time.shape) # initial conditions
                f_m = f_s*np.ones(time.shape) # initial conditions
                z_b = (1+0.0j)*np.ones(time.shape) # initial conditions
                f_b = f_s*np.ones(time.shape) # initial conditions
                for n, t in enumerate(time[:-1]):
                    
                    z_m[n+1] = z_m[n] + T*f_m[n]*(z_m[n]*(a_m + 1j*2*np.pi + b_1m*np.power(np.abs(z_m[n]),2) + b_2m*np.power(np.abs(z_m[n]),4)/(1 - np.power(np.abs(z_m[n]),2))) + x[n])
                    f_m[n+1] = f_m[n] + T*f_m[n]*(-l1*np.real(x[n])*np.sin(np.angle(z_m[n])) - (l2/10)*(np.power(base,(f_m[n]-f_b[n])/base)-1))
                    z_b[n+1] = z_b[n] + T*f_b[n]*(z_b[n]*(a_m + 1j*2*np.pi + b_1b*np.power(np.abs(z_b[n]),2)) + np.exp(1j*np.angle(z_m[n])))
                    f_b[n+1] = f_b[n] + T*f_b[n]*(-l1*np.cos(np.angle(z_m[n]))*np.sin(np.angle(z_b[n])) - l2*(np.power(base,(f_b[n]-f_0)/base)-1))
            
                peaks, _ = find_peaks(np.real(z_b))
                peaks = 1000*peaks/fs # converting to miliseconds
                peaks = peaks[:128]
                slope, _, _, _, _ = linregress(range(len(np.diff(peaks))), np.diff(peaks))
                if f_s == f_0:
                    mean_slope = slope
                else:
                    bar_results[if_s - 1] = slope - mean_slope

                print(np.diff(peaks))

            print(l1,l2)
            print(bar_results)
            ax = plt.subplot(gs[il1, il2])
            ax.bar(range(len(f_stim)-1), bar_results)
            ax.grid(linestyle='dashed')
            ax.set_axisbelow(True)
            #ax.set_ylim([-0.5, 0.5])
            if il2 == 0:
                ax.set_ylabel(r'$\lambda_1 = {}$'.format(l1))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d ms'))
            else:
                ax.set_yticklabels([])
            if il1 == (len(l1s) - 1):
                ax.set_xlabel(r'$\lambda_2 = {}$'.format(l2))
                ax.set_xticks(range(len(f_stim)-1))
                ax.set_xticklabels(['Faster','Fast','Slow','Slower'])
            else:
                ax.set_xticks(range(len(f_stim)-1))
                ax.set_xticklabels([])
    
    plt.show()
