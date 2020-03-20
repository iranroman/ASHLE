import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
from human_data import get_zamm_etal_2018_data

# time parameters
fs = 500
T = 1/fs
dur = 50
time = np.arange(0,dur,T)

# oscillator parameters
a = 1
b = -1
l1 = 2.5 # learning rate
l2 = 0.00002 # elasticity
nlearn = 8 # number of metronome learning beats
z = (1.0+0.0j)*np.ones(time.shape) # initial conditions

# human data (Zamm et al. 2018)
subjs_data = get_zamm_etal_2018_data()
#subjs_data = [[450, 250, 350, 600, 800]]
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

        for n, t in enumerate(time[:-1]):
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + x[n])
            f[n+1] = f[n] + T*(-f[n]*l1*np.real(x[n])*np.sin(np.angle(z[n])) - l2*(-np.power(spf,4)+np.power(f[n],4)))#np.power(np.abs((spf)-(f[n])),2)*(spf-f[n])/spf)

        #plt.plot(time,np.real(z))
        #plt.plot(time,np.real(x))
        #plt.plot(time,1/f)
        #plt.show()
        #plt.close()
        print('###############################')
        print('stimulus IOI (Hz): ', 1000/f0)
        print('learned IOI (ms): ', 1000/f[int(nlearn*fs/f0)])
        peaks, _ = find_peaks(np.real(z[int((nlearn+1.5)*fs/f0):]))
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
        #plt.plot(np.diff(peaks))
        #plt.show()
        #plt.close()

    allslopes.append(slopes)

allslopes = np.asarray(allslopes)
mean_slopes = np.mean(allslopes,0)
SE_slopes = np.std(allslopes,0)/np.sqrt(allslopes.shape[0])

plt.subplot(2,2,1)
plt.bar(np.arange(4),[0.07,0.03,-0.025,-0.19],yerr=[0.015,0.015,0.02,0.02])
plt.grid()
plt.ylim([-0.3, 0.15])
plt.subplot(2,2,2)
plt.bar(np.arange(4),mean_slopes,yerr=SE_slopes)
plt.grid()
plt.ylim([-0.3, 0.15])
plt.subplot(2,2,4)
plt.scatter(np.ones(allslopes.shape[0]),allslopes[:,0])
plt.scatter(2*np.ones(allslopes.shape[0]),allslopes[:,1])
plt.scatter(3*np.ones(allslopes.shape[0]),allslopes[:,2])
plt.scatter(4*np.ones(allslopes.shape[0]),allslopes[:,3])
plt.grid()
plt.ylim([-1.25, 0.5])
plt.savefig('../figures_raw/fig2.eps')
