import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
from human_data import get_zamm_etal_2018_data_and_result

# time parameters
fs = 500
T = 1/fs
dur = 150
time = np.arange(0,dur,T)

# oscillator parameters
a_d = 1
b_d = -1
b2_d = 0
F_d = 1
a = 1
b = -1
l1 = 4 # learning rate
l2 = 1.4 # 0.00019 # elasticity
l2_d = l2/100 # 0.00019 # elasticity
base=np.exp(1)
nlearn = 0 # number of metronome learning beats
z = (1.0+0.0j)*np.ones(time.shape) # initial conditions
d = (0.99+0.0j)*np.ones(time.shape) # initial conditions

# human data and results (Zamm et al. 2018)
zamm_etal_2018 = get_zamm_etal_2018_data_and_result()
subjs_data = zamm_etal_2018['data']
#subjs_data = [[440, 256, 331, 572, 800]]
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
        f = f0*np.ones(time.shape)
        f_d = f0*np.ones(time.shape)

        for n, t in enumerate(time[:-1]):
            d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2)) + b2_d*np.power(np.abs(d[n]),4)/(1-np.power(np.abs(d[n]),2))) + F_d*x[n])
            f_d[n+1] = f_d[n] + T*f_d[n]*(-l1*np.real(F_d*x[n])*np.sin(np.angle(d[n])) - l2_d*(np.power(base,(f_d[n]-f[n])/base)-1))
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + np.exp(1j*np.angle(d[n])))
            f[n+1] = f[n] + T*f[n]*(-l1*np.cos(np.angle(d[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f[n]-spf)/base)-1))

        #plt.subplot(2,1,1)
        #plt.plot(np.real(z[:]))
        #plt.plot(np.real(x[:]))
        #plt.plot(1/f[:])
        #plt.grid()
        #plt.subplot(2,1,2)
        #plt.plot(np.real(d[:]))
        #plt.plot(np.real(x[:]))
        #plt.plot(1/f_d[:])
        #plt.grid()
        #plt.show()
        print('###############################')
        print('stimulus IOI (Hz): ', 1000/f0)
        print('learned IOI (ms): ', 1000/f[int((nlearn)*fs/f0)])
        peaks, _ = find_peaks(np.real(z[int((nlearn)*fs/f0):]))
        peaks = peaks[:128]
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
