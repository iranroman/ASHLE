import sys
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
f_stim = [1.5, 1.3, 1.15, 0.85, 0.7, 0.5]
l1s = [0.3, 0.4, 0.5, 0.6, 0.7] 
l2s = [0.0, 0.04, 0.05, 0.06, 0.07] 
base=2
nlearn = 4

numplots = len(l1s)*len(l2s)
plt.figure(figsize=(12,16))
gs = gridspec.GridSpec(len(l1s), len(l2s), wspace=0, hspace=0)

for il1, l1 in enumerate(l1s):
    for il2, l2 in enumerate(l2s):

        bar_results = np.zeros((len(f_stim)))
        for if_s, f_s in enumerate(f_stim):

            x = np.exp(1j*2*np.pi*time*f_s)
            z_m = (0+0.0j)*np.ones(time.shape) # initial conditions
            f_m = np.ones(time.shape) # initial conditions
            z_b = (0+0.0j)*np.ones(time.shape) # initial conditions
            f_b = np.ones(time.shape) # initial conditions
            for n, t in enumerate(time[:-1]):
                
                z_m[n+1] = z_m[n] + T*f_m[n]*(z_m[n]*(a_m + 1j*2*np.pi + b_1m*np.power(np.abs(z_m[n]),2) + b_2m*np.power(np.abs(z_m[n]),4)/(1 - np.power(np.abs(z_m[n]),2))) + x[n])
                f_m[n+1] = f_m[n] + T*f_m[n]*(-l1*np.real(x[n])*np.sin(np.angle(z_m[n])) - l2*(np.power(base,f_m[n]) - np.power(base,f_b[n]))/base)
                z_b[n+1] = z_b[n] + T*f_b[n]*(z_b[n]*(a_m + 1j*2*np.pi + b_1b*np.power(np.abs(z_b[n]),2)) + np.exp(1j*np.angle(z_m[n])))
                f_b[n+1] = f_b[n] + T*f_b[n]*(-l1*np.cos(np.angle(z_m[n]))*np.sin(np.angle(z_b[n])) - l2*(np.power(base,f_b[n]) - np.power(base,f_0))/base)
        
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
                bar_results[if_s] = 1000*np.mean((locs_z[0:-2] - locs_x[mid_F_peaks_index:pen_F_peaks_index])/fs) 

        ax = plt.subplot(gs[il1, il2])
        ax.bar(range(len(f_stim)), bar_results)
        ax.grid(linestyle='dashed')
        ax.set_axisbelow(True)
        ax.set_ylim([-25, 25])
        if il2 == 0:
            ax.set_ylabel(r'$\lambda_1 = {}$'.format(l1))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d ms'))
        else:
            ax.set_yticklabels([])
        if il1 == (len(l1s) - 1):
            ax.set_xlabel(r'$\lambda_2 = {}$'.format(l2))
            ax.set_xticks(range(len(f_stim)))
            ax.set_xticklabels(['F50','F30','F15','S15','S30','S50'])
        else:
            ax.set_xticklabels([])

plt.show()


# human data and results (Zamm et al. 2018)
zamm_etal_2018 = get_zamm_etal_2018_data_and_result()
subjs_data = zamm_etal_2018['data']
#subjs_data = [[420, 250, 350, 600, 800]]
result = zamm_etal_2018['result']

# simulations
allslopes = []
for subj_data in subjs_data:

    spr = subj_data[0]
    spf  = 1000/spr

    mean_slope = 0
    spr_cv = 0
    slopes = []
    cvs = []
    for pr in subj_data:

        f0 = 1000/pr
        x = np.exp(1j*2*np.pi*time*f0)
        x[int(nlearn*fs/f0):] = 0
        f = (spf+0.01*np.random.randn())*np.ones(time.shape)
        f_d = (spf+0.01*np.random.randn())*np.ones(time.shape)

        for n, t in enumerate(time[:-1]):
            d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2)) + b2_d*np.power(np.abs(d[n]),4)/(1-np.power(np.abs(d[n]),2))) + F_d*x[n])
            f_d[n+1] = f_d[n] + T*(-f_d[n]*l1*np.real(F_d*x[n])*np.sin(np.angle(d[n])) - l2_d*(np.power(base,f_d[n])-np.power(base,f[n]))/base)
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + d[n])
            f[n+1] = f[n] + T*(-f[n]*l1*np.real(d[n])*np.sin(np.angle(z[n])) - l2*(np.power(base,f[n]) - np.power(base,spf))/base)

        #plt.subplot(2,1,1)
        #plt.plot(np.real(z[:14000]))
        #plt.plot(np.real(x[:14000]))
        #plt.plot(1/f[:14000])
        #plt.grid()
        #plt.subplot(2,1,2)
        #plt.plot(np.real(d[:14000]))
        #plt.plot(np.real(x[:14000]))
        #plt.plot(1/f_d[:14000])
        #plt.grid()
        #plt.show()
        print('###############################')
        print('stimulus IOI (Hz): ', 1000/f0)
        print('learned IOI (ms): ', 1000/f[int((nlearn+10)*fs/f0)])
        peaks, _ = find_peaks(np.real(z[int((nlearn+1.5)*fs/f0):]))
        peaks = peaks[0:]
        peaks = 1000*peaks/fs # converting to miliseconds
        slope, _, _, _, _ = linregress(range(len(np.diff(peaks))), np.diff(peaks))
        cv = np.std(np.diff(peaks))/np.mean(np.diff(peaks))
        print('Slope: ', slope - mean_slope)
        print('CV: ', cv)
        if pr == spr:
            mean_slope = slope
            spr_cv = cv
        else:
            slopes.append(slope-mean_slope)
            cvs.append(cv)

    allslopes.append(slopes)

# analysis
allslopes = np.asarray(allslopes)
mean_slopes = np.mean(allslopes,0)
SE_slopes = np.std(allslopes,0)/np.sqrt(allslopes.shape[0])

# figure
barWidth = 0.4
barLoc = 0.2
loc1 = np.arange(len(mean_slopes))-barLoc
loc2 = [x + barWidth for x in loc1]
plt.bar(loc1,result[0],yerr=result[1],width=barWidth,color='silver',capsize=5,edgecolor='black',label='Musicians')
plt.bar(loc2,mean_slopes,yerr=SE_slopes,width=barWidth,color='lightskyblue',capsize=5,edgecolor='black',label='ASHLE')
plt.grid()
plt.ylim([-0.3, 0.15])
plt.axhline(0,color='black')
plt.xlabel('Rate Condition')
plt.xticks(range(len(mean_slopes)),['Faster','Fast','Slow','Slower'])
plt.ylabel('Adjusted Mean Slope of IOIs')
plt.legend(loc='lower left')
plt.savefig('../figures_raw/fig2.eps')
