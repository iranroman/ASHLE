import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
from human_data import get_zamm_etal_2018_data

# time parameters
fs = 1000
T = 1/fs
dur = 25
time = np.arange(0,dur,T)

# oscillator parameters
a = 1
b = -1
l1 = 5 # learning rate
l2 = 0.0025 # elasticity
nlearn = 8 # number of metronome learning beats
z = (1.0+0.0j)*np.ones(time.shape) # initial conditions

# human data (Zamm et al. 2018)
sprs = get_zamm_etal_2018_data()

sprs = np.array(sprs)
averages = np.mean(sprs,0)

mean_slope = 0
spr_cv = 0
slopes = []
cvs = []

    
for i, f_s in enumerate([spr, faster, fast, slow, slower]):

    x = np.exp(1j*2*np.pi*time*f_s)
    x[int(nlearn*fs/f_s):] = 0
    f = (spr+0.01*np.random.randn())*np.ones(time.shape)

    for n, t in enumerate(time[:-1]):
        z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + x[n])
        f[n+1] = f[n] + T*(-np.power(np.abs(f[n]-f_s),2)*l1*(np.real(x[n]))*np.sin(np.angle(z[n])) - (np.power(np.abs(spr-f[n]),2))*l2*(f[n]-spr)/spr)

    print('###############################')
    print('stimulus freq (Hz): ',f_s)
    print('learned IOI (ms): ', 1000/f[int(nlearn*fs/f_s)])
    #plt.plot(time,np.real(z))
    #plt.plot(time,np.real(x))
    #plt.plot(time,1/f)
    #plt.show()
    #plt.close()
    peaks, _ = find_peaks(np.real(z[int((nlearn+1.5)*fs/f_s):]))
    peaks = 1000*peaks/fs # converting to miliseconds
    slope, _, _, _, _ = linregress(range(len(np.diff(peaks))), np.diff(peaks))
    cv = np.std(np.diff(peaks))/np.mean(np.diff(peaks))
    print('Slope: ', slope - mean_slope)
    print('CV: ', cv)
    #plt.plot(np.diff(peaks))
    #plt.show()
    #plt.close()
    if i == 0:
        mean_slope = slope
        spr_cv = cv
    else:
        slopes.append(slope-mean_slope)
        cvs.append(cv)

cvs.insert(2,spr_cv)
plt.subplot(1,2,1)
plt.bar(range(5),cvs)
plt.subplot(1,2,2)
plt.bar(range(4),slopes)
plt.show()

plt.bar(range(4),averages[1:],yerr=np.std(sprs[:,1:],0)/np.sqrt(sprs.shape[0]))
plt.ylim([0,1000])
plt.savefig('../figures_raw/fig2.eps', format='eps')
