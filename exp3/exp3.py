import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from math import pi

# Simulation parameters
fs        = 10000
T         = 1/fs
dur       = 38
t         = np.linspace(0, dur, dur*fs)
ntime     = t.size
halfsamps = np.floor(ntime/2)


# z - parameters
a_d = 1
b_d = -1
b2_d = 0
F_d = 1.0 #0.03
a = 1
b = -100
#%%%%%%%%%%%%%%%%%% HEBBIAN LEARNING PARAMETERS %%%%%%%%%%%%%%
l1 = 4 # learning rate
l2 = 2 # 0.00019 # elasticity
l2_d = l2/50 # 0.00019 # elasticity
gsigm = 0.01 #7.5 #15
sigma = 0.0006
base=np.exp(1)

z = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator1
dz = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.size)
y = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator2
dy = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.size) 

f_1 = np.zeros(t.shape)  # Adaptive frequency osc1
f_d1 = np.zeros(t.shape)  # Adaptive frequency osc1
f_2 = np.zeros(t.shape)  # Adaptive frequency osc2
f_d2 = np.zeros(t.shape)  # Adaptive frequency osc2

#%%%%%%%%%%%%% Group Mismatch - SPR diff > 110 ms %%%%%%%%%%%%%%%%%%%%%%%%
freqs_miss1 = np.array([180, 210, 215, 220, 280, 310, 330, 350, 340, 420])
freqs_miss2 = np.array([310, 340, 330, 380, 410, 460, 462, 470, 500, 520])

#%%%%%%%%%%%%% Group Match - SPR diff < 10 ms %%%%%%%%%%%%%%%%%%%%%%%%%%%%
freqs_match1 = np.array([269, 280, 350, 343, 373, 376, 398, 420, 433, 451])
freqs_match2 = np.array([278, 289, 359, 352, 380, 384, 407, 429, 440, 460])

f_s = 1000/400                       # Metronome Frequency 
F = 0.1*np.exp(1j * 2 * pi * t * (f_s))  # Stimulus "Metronome"

mean_SPR_miss_pairs  = np.zeros((freqs_miss1.size, 4)) 
mean_SPR_match_pairs = np.zeros((freqs_miss1.size, 4))

tr_time = 0.400 * 4        # training time (s)
tr_samps = int(tr_time * fs)  # training samples

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    # millisenconds to Hz conversion
    f_1[0] = 1000/freqs_miss1[i]
    f_d1[0] = 1000/freqs_miss1[i]
    f_2[0] = 1000/freqs_miss2[i] 
    f_d2[0] = 1000/freqs_miss2[i]
    f_1_0 = 1000/freqs_miss1[i]
    f_2_0 = 1000/freqs_miss2[i] 
        
    # Forward Euler integration
    for n in range(ntime-1):
        if t[n] <= tr_time:

            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F[n])
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*(np.real(F[n])*np.sin(np.angle(z[n]))-np.imag(F[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(y[n]),2))) + F[n])
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*(np.real(F[n])*np.sin(np.angle(y[n]))-np.imag(F[n])*np.cos(np.angle(y[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))

        else:

            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*y[n] + (1/(z[n]-y[n]))*(np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1)))
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*(np.real(F_d*y[n])*np.sin(np.angle(z[n]))-np.imag(F_d*y[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(y[n]),2))) + F_d*z[n] + (1/(y[n]-z[n]))*(np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1)))
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*(np.real(F_d*z[n])*np.sin(np.angle(y[n]))-np.imag(F_d*z[n])*np.cos(np.angle(y[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))


    zpeaks, _ = find_peaks(np.real(z[tr_samps:]), prominence=0.1)
    ypeaks, _ = find_peaks(np.real(y[tr_samps:]), prominence=0.1)

    print(f_1_0, f_2_0)

    if np.abs(len(zpeaks) - len(ypeaks)) > 2:
        raise ValueError('Different number of peaks between Models')
	
    lag_mat = np.zeros((64,10))
    for j in range(10):
        lag_mat[:,j]=np.abs(zpeaks[4:4+64] - ypeaks[j:j+64])
    corr_idx = np.argmin(np.mean(lag_mat,0))
    peaksdiff = np.reshape(lag_mat[:64,corr_idx],(16,4))
    mean_SPR_miss_pairs[i,:] = np.mean(peaksdiff,0)/fs
    print(mean_SPR_miss_pairs)
print(np.mean(mean_SPR_miss_pairs,0))

                  

f_1 = (1000/400)*np.ones(t.shape)  # Adaptive frequency osc1
f_d1 = (1000/400)*np.ones(t.shape) # Adaptive frequency osc1
f_2 = (1000/400)*np.ones(t.shape)  # Adaptive frequency osc2
f_d2 = (1000/400)*np.ones(t.shape) # Adaptive frequency osc2

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_match1[i]
    f_d1[0] = 1000/freqs_match1[i]
    f_2[0] = 1000/freqs_match2[i]
    f_d2[0] = 1000/freqs_match2[i]
    f_1_0 = 1000/freqs_match1[i]
    f_2_0 = 1000/freqs_match2[i] 
        
    # Forward Euler integration
    for n in range(ntime-1):
        if t[n] <= tr_time:

            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F[n])
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*(np.real(F[n])*np.sin(np.angle(z[n]))-np.imag(F[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(y[n]),2))) + F[n])
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*(np.real(F[n])*np.sin(np.angle(y[n]))-np.imag(F[n])*np.cos(np.angle(y[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))

        else:

            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*y[n] + (1/(z[n]-y[n]))*(np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1)))
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*(np.real(F_d*y[n])*np.sin(np.angle(z[n]))-np.imag(F_d*y[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(y[n]),2))) + F_d*z[n] + (1/(y[n]-z[n]))*(np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1)))
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*(np.real(F_d*z[n])*np.sin(np.angle(y[n]))-np.imag(F_d*z[n])*np.cos(np.angle(y[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))

    zpeaks, _ = find_peaks(np.real(z[tr_samps:]), prominence=0.1)
    ypeaks, _ = find_peaks(np.real(y[tr_samps:]), prominence=0.1)

    if np.abs(len(zpeaks) - len(ypeaks)) > 3:
        raise ValueError('Different number of peaks between Models')

    lag_mat = np.zeros((64,10))
    for j in range(10):
        lag_mat[:,j]=np.abs(zpeaks[4:4+64]+0*np.random.normal(0, 12, 64) - ypeaks[j:j+64]+0*np.random.normal(0, 12, 64))
    corr_idx = np.argmin(np.mean(lag_mat,0))
    peaksdiff = np.reshape(lag_mat[:64,corr_idx],(16,4))
    mean_SPR_match_pairs[i,:] = np.mean(peaksdiff,0)/fs
    print(mean_SPR_match_pairs)
print(np.mean(mean_SPR_match_pairs,0))

ind = np.arange(len(np.mean(mean_SPR_match_pairs, 0))) # x locations for the groups
width = 0.35  # width of the bars

# # Plot fig5.eps
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.bar(ind-width/2, 1000*np.mean(mean_SPR_match_pairs, 0), width, bottom=0,color='grey',edgecolor='black',label='Match')
# ax.errorbar(ind-width/2, 1000*np.mean(mean_SPR_match_pairs, 0), 1000*np.std(mean_SPR_match_pairs)/np.sqrt(mean_SPR_match_pairs.shape[0]), [0,0,0,0],'none',ecolor='black')
# ax.bar(ind+width/2, 1000*np.mean(mean_SPR_miss_pairs, 0), width, bottom=0, color='white',edgecolor='black',label='Mismatch')
# ax.errorbar(ind+width/2, 1000*np.mean(mean_SPR_miss_pairs, 0), 1000*np.std(mean_SPR_miss_pairs)/np.sqrt(mean_SPR_miss_pairs.shape[0]), [0,0,0,0],'none',ecolor='black')
# ax.set_ylim([0, 35])
# ax.set_xlim([-0.5, 3.5])
# ax.set_xticks(np.arange(4))
# ax.set_xticklabels(['1','2','3','4'],fontsize=15)
# ax.set_xlabel('Melody repetition',fontsize=15)
# ax.grid(color='gray', linestyle='dashed')
# ax.set_axisbelow(True)
# ax.tick_params(axis="y", labelsize=15)
# ax.tick_params(axis="x", labelsize=15)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.legend(loc='upper right',prop={'size': 13})
# ax.set_ylabel('Mean absolute asynchrony (ms)',fontsize=15)
# plt.savefig('../figures_raw/fig5.eps')

# Plot fig4.eps
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
ax1.bar(ind-width/2, [17.5, 16.5, 16.3, 17.2], width, bottom=0,color='grey',edgecolor='black',label='Match')
ax1.errorbar(ind-width/2, [17.5, 16.5, 16.3, 17.2], [0.8, 0.6, 0.7, 1], [0,0,0,0],'none',ecolor='black')
ax1.bar(ind+width/2, [24.7, 21.5, 21, 21.9], width, bottom=0, color='white',edgecolor='black',label='Mismatch')
ax1.errorbar(ind+width/2, [24.7, 21.5, 21, 21.9], [2.3, 1.5, 1.5, 1.5], [0,0,0,0],'none',ecolor='black')
ax1.set_ylim([0, 35])
ax1.set_xlim([-0.5, 3.5])
ax1.set_ylabel('Mean absolute asynchrony (ms)',fontsize=15)
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(['1','2','3','4'],fontsize=15)
ax1.set_xlabel('Melody repetition',fontsize=15)
ax1.grid(color='gray', linestyle='dashed')
ax1.set_axisbelow(True)
ax1.tick_params(axis="y", labelsize=15)
ax1.tick_params(axis="x", labelsize=15)
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=25, weight='bold')
ax1.legend(loc='upper right',prop={'size': 13})
ax2.bar(ind-width/2, 1000*np.mean(mean_SPR_match_pairs, 0), width, bottom=0,color='grey',edgecolor='black',label='Match')
ax2.errorbar(ind-width/2, 1000*np.mean(mean_SPR_match_pairs, 0), 1000*np.std(mean_SPR_match_pairs)/np.sqrt(mean_SPR_match_pairs.shape[0]), [0,0,0,0],'none',ecolor='black')
ax2.bar(ind+width/2, 1000*np.mean(mean_SPR_miss_pairs, 0), width, bottom=0, color='white',edgecolor='black',label='Mismatch')
ax2.errorbar(ind+width/2, 1000*np.mean(mean_SPR_miss_pairs, 0), 1000*np.std(mean_SPR_miss_pairs)/np.sqrt(mean_SPR_miss_pairs.shape[0]), [0,0,0,0],'none',ecolor='black')
ax2.set_ylim([0, 35])
ax2.set_xlim([-0.5, 3.5])
ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(['1','2','3','4'],fontsize=15)
ax2.set_xlabel('Melody repetition',fontsize=15)
ax2.grid(color='gray', linestyle='dashed')
ax2.set_axisbelow(True)
ax2.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="x", labelsize=15)
ax2.yaxis.grid(color='gray', linestyle='dashed')
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=25, weight='bold')
ax2.legend(loc='upper right',prop={'size': 13})

f_1 = (1000/400)*np.ones(t.shape[0]//2);  # Adaptive frequency osc1
f_d1 = (1000/400)*np.ones(t.shape[0]//2);  # Adaptive frequency osc1
f_2 = (1000/400)*np.ones(t.shape[0]//2);  # Adaptive frequency osc2
f_d2 = (1000/400)*np.ones(t.shape[0]//2);  # Adaptive frequency osc2
z = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.shape[0]//2) # oscillator1
dz = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.shape[0]//2)
y = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.shape[0]//2) # oscillator2
dy = 0.0 * np.exp(1j * 2 * pi) * np.ones(t.shape[0]//2) 

nsprs = 5
mean_asynchs = np.zeros((4, nsprs,6))
all_sprs = np.linspace(350,650,nsprs)
for ispr, spr in enumerate(all_sprs):

    for jspr, pair_spr in enumerate([-220, -110, -10, 10, 110, 220]):
        
        for irepeat in range(4):
    
            f_1[0] = 1000/spr
            f_d1[0] = 1000/spr
            f_2[0] = 1000/(spr + pair_spr)
            f_d2[0] = 1000/(spr + pair_spr)
            f_1_0 = 1000/spr
            f_2_0 = 1000/(spr + pair_spr) 
                
            # Forward Euler integration
            for n in range((ntime//2)-1):
                if t[n] <= tr_time:

                    z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F[n])
                    f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*(np.real(F[n])*np.sin(np.angle(z[n]))-np.imag(F[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))
        
                    y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(y[n]),2))) + F[n])
                    f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*(np.real(F[n])*np.sin(np.angle(y[n]))-np.imag(F[n])*np.cos(np.angle(y[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))
        
                else:
        
                    z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + F_d*y[n] + (1/(z[n]-y[n]))*(np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1)))
                    f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*(np.real(F_d*y[n])*np.sin(np.angle(z[n]))-np.imag(F_d*y[n])*np.cos(np.angle(z[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))
        
                    y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(y[n]),2))) + F_d*z[n] + (1/(y[n]-z[n]))*(np.random.normal(0,gsigm,1) + 1j*np.random.normal(0,gsigm,1)))
                    f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*(np.real(F_d*z[n])*np.sin(np.angle(y[n]))-np.imag(F_d*z[n])*np.cos(np.angle(y[n]))) - np.abs(F_d+np.random.normal(0,sigma,1))*l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))

            zpeaks, _ = find_peaks(np.real(z[tr_samps:]), prominence=0.1)
            ypeaks, _ = find_peaks(np.real(y[tr_samps:]), prominence=0.1)
    
            if np.abs(len(zpeaks) - len(ypeaks)) > 2:
                mean_asynchs[:,ispr,jspr] = np.asarray([np.nan]*4)
                print(np.nan)
                break
        
            lag_mat = np.zeros((16,6))

            for j in range(6):
                lag_mat[:,j]=np.abs(zpeaks[3:3+16]+0*np.random.normal(0, 12, 16) - ypeaks[j:j+16]+0*np.random.normal(0, 12, 16))
            corr_idx = np.argmin(np.mean(lag_mat,0))
            peaksdiff = lag_mat[:16,corr_idx]
            mean_asynchs[irepeat,ispr,jspr] = np.mean(peaksdiff,0)/fs
            print(np.mean(peaksdiff,0)/fs)
        print(np.mean(mean_asynchs,0))

allslopes = np.mean(mean_asynchs,0)
for i in range(len(all_sprs)):
    ax3.plot(range(6),1000*allslopes[i],'-o',linewidth=3,markersize=6,label=str(int(all_sprs[i]))+'ms')
ax3.legend(loc='lower left',prop={'size': 11})
ax3.set_ylim([0, 50])
ax3.set_xlim([-0.5, 5.5])
ax3.set_axisbelow(True)
ax3.grid(color='gray', linestyle='dashed')
ax3.set_xticks(np.arange(6))
ax3.tick_params(axis="y", labelsize=15)
ax3.set_xticklabels(['-220','-110','-10','+10','+110','+220'],fontsize=12)
ax3.set_xlabel('Partner SMT difference (ms)',fontsize=15)
ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, size=25, weight='bold')
plt.savefig('../figures_raw/fig4.eps')
