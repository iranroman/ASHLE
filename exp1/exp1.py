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
F_d = 0
a = 1
b = -100
l1 = 4 # learning rate
l2 = 2 # 0.00019 # elasticity
l2_d = l2/50 # 0.00019 # elasticity
base = np.exp(1)
gsigm = 0
sigma = 0.0006
nlearn = 0 # number of metronome learning beats
z = (0.001+0.0j)*np.ones(time.shape) # initial conditions
d = (0.001+0.0j)*np.ones(time.shape) # initial conditions

# human data and results (Zamm et al. 2018)
zamm_etal_2018 = get_zamm_etal_2018_data_and_result()
subjs_data = zamm_etal_2018['data']
#subjs_data = [subjs_data[-1]]
#subjs_data = [[440, 256, 331, 572, 790]]
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
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*x[n] + np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1))
            f[n+1] = f[n] + T*f[n]*(-l1*(np.real(F_d*x[n])*np.sin(np.angle(z[n])) - np.imag(F_d*x[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f[n]-spf)/spf)-1))
            #d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2)) + b2_d*np.power(np.abs(d[n]),4)/(1-np.power(np.abs(d[n]),2))))
            #f_d[n+1] = f_d[n] + T*f_d[n]*(-(l1/((spf)*600))*np.cos(np.angle(z[n]))*np.sin(np.angle(d[n])))
            #z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + np.exp(1j*np.angle(d[n])))
            #f[n+1] = f[n] + T*f[n]*(-l1*np.cos(np.angle(d[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f[n]-spf)/spf)-1))

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

fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True,figsize=(15,5))
ax1.bar(np.arange(4),result[0],color='grey',edgecolor='black')
ax1.errorbar(np.arange(4),result[0],result[1],[0,0,0,0],'none',ecolor='black')
ax1.set_ylim([-0.3, 0.2])
ax2.set_xlim([-0.5, 3.5])
ax1.set_ylabel('Mean adjusted slope of IOIs',fontsize=15)
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(['Faster','Fast','Slow','Slower'],fontsize=15)
ax1.grid(color='gray', linestyle='dashed')
ax1.set_axisbelow(True)
ax1.tick_params(axis="y", labelsize=15)
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=25, weight='bold')
ax2.bar(np.arange(len(mean_slopes)), mean_slopes,color='grey',edgecolor='black')
ax2.errorbar(np.arange(len(mean_slopes)), mean_slopes, SE_slopes,[0,0,0,0],'none',ecolor='black')
ax2.set_ylim([-0.3, 0.2])
ax2.set_xlim([-0.5, 3.5])
ax2.set_axisbelow(True)
ax2.grid(color='gray', linestyle='dashed')
ax2.set_xticks(np.arange(len(mean_slopes)))
ax2.set_xticklabels(['Faster','Fast','Slow','Slower'],fontsize=15)
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=25, weight='bold')


subjs_data = np.linspace(350,650,5)
#subjs_data = [350, 650]
result = zamm_etal_2018['result']

# simulations
allslopes = []
for subj_data in subjs_data:

    spr = subj_data
    spf  = 1000/spr

    mean_slope = 0
    spr_cv = 0
    slopes = []
    cvs = []
    for pr in [1, 0.55, 0.7, 0.85, 1.15, 1.3, 1.45]:

        f0 = 1000/(spr*pr)
        x = np.exp(1j*2*np.pi*time*f0)
        x[int(nlearn*fs/f0):] = 0
        f = f0*np.ones(time.shape)
        f_d = f0*np.ones(time.shape)

        for n, t in enumerate(time[:-1]):
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*x[n] + np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1))
            f[n+1] = f[n] + T*f[n]*(-l1*(np.real(F_d*x[n])*np.sin(np.angle(z[n])) - np.imag(F_d*x[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f[n]-spf)/spf)-1))
            #z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*x[n] + np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1))
            #f[n+1] = f[n] + T*f[n]*(-l1*np.real(F_d*x[n])*np.sin(np.angle(z[n])) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f[n]-spf)/spf)-1))
            #d[n+1] = d[n] + T*f_d[n]*(d[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(d[n]),2)) + b2_d*np.power(np.abs(d[n]),4)/(1-np.power(np.abs(d[n]),2))))
            #f_d[n+1] = f_d[n] + T*f_d[n]*(-(l1/((spf)*600))*np.cos(np.angle(z[n]))*np.sin(np.angle(d[n])))
            #z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + np.exp(1j*np.angle(d[n])))
            #f[n+1] = f[n] + T*f[n]*(-l1*np.cos(np.angle(d[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f[n]-spf)/spf)-1))

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
        if pr == 1:
            mean_slope = slope
            spr_cv = cv
        else:
            slopes.append(slope-mean_slope)
            cvs.append(cv)

    allslopes.append(slopes)

allslopes = np.asarray(allslopes)
for i in range(len(subjs_data)):
    ax3.plot(range(6),allslopes[i],'-o',linewidth=3,markersize=6,label=str(int(subjs_data[i]))+'ms')
ax3.legend(loc='lower left',prop={'size': 13})
ax3.set_ylim([-0.3, 0.2])
ax3.set_xlim([-0.5, 5.5])
ax3.set_axisbelow(True)
ax3.grid(color='gray', linestyle='dashed')
ax3.set_xticks(np.arange(6))
ax3.set_xticklabels(['F45','F30','F15','S15','S30','S45'],fontsize=15)
ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, size=25, weight='bold')
plt.savefig('../figures_raw/fig3.eps')
