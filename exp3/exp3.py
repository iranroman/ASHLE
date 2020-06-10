import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from math import pi

# Simulation parameters
fs        = 1000
T         = 1/fs
dur       = 33
t         = np.linspace(0, dur, dur*fs)
ntime     = t.size
halfsamps = np.floor(ntime/2);


# z - parameters
a_d = 1
b_d = -1
b2_d = 0
F_d = 0.03
a = 1
b = -1
#%%%%%%%%%%%%%%%%%% HEBBIAN LEARNING PARAMETERS %%%%%%%%%%%%%%
l1 = 5 # learning rate
l2 = 1.4 # 0.00019 # elasticity
l2_d = l2/50 # 0.00019 # elasticity
base=np.exp(1)

z = 1.0 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator1
dz = 0.99 * np.exp(1j * 2 * pi) * np.ones(t.size)
y = 1.0 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator2
dy = 0.99 * np.exp(1j * 2 * pi) * np.ones(t.size) 

f_1 = np.zeros(t.shape);  # Adaptive frequency osc1
f_d1 = np.zeros(t.shape);  # Adaptive frequency osc1
f_2 = np.zeros(t.shape);  # Adaptive frequency osc2
f_d2 = np.zeros(t.shape);  # Adaptive frequency osc2

#%%%%%%%%%%%%% Group Mismatch - SPR diff > 110 ms %%%%%%%%%%%%%%%%%%%%%%%%
freqs_miss1 = np.array([180, 210, 215, 220, 280, 310, 330, 350, 340, 420])
freqs_miss2 = np.array([310, 340, 330, 380, 410, 460, 462, 470, 500, 520])

#%%%%%%%%%%%%% Group Match - SPR diff < 10 ms %%%%%%%%%%%%%%%%%%%%%%%%%%%%
freqs_match1 = np.array([269, 280, 350, 343, 373, 376, 398, 420, 433, 451])
freqs_match2 = np.array([278, 289, 359, 352, 280, 384, 407, 429, 440, 460])

f_s = 1000/400                       # Metronome Frequency 
F = np.exp(1j * 2 * pi * t * (f_s))  # Stimulus "Metronome"

mean_SPR_miss_pairs  = np.zeros((freqs_miss1.size, 4)) 
mean_SPR_match_pairs = np.zeros((freqs_miss1.size, 4))

tr_time = 0.400 * 4        # training time (s)
tr_samps = int(tr_time * fs)  # training samples

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_miss1[i]
    f_d1[0] = 1000/freqs_miss1[i]
    f_2[0] = 1000/freqs_miss2[i] 
    f_d2[0] = 1000/freqs_miss2[i]
    f_1_0 = 1000/freqs_miss1[i]
    f_2_0 = 1000/freqs_miss2[i] 
        
    # Forward Euler integration
    for n in range(ntime-1):
        if t[n] <= tr_time:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F[n])
            f_d1[n+1] = f_d1[n] + T*f_d1[n]*(-l1*np.real(F[n])*np.sin(np.angle(dz[n])) - l2_d*(f_d1[n]-f_1[n])/f_1[n])
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + np.exp(1j*np.angle(dz[n])))
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*np.cos(np.angle(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F[n])
            f_d2[n+1] = f_d2[n] + T*f_d2[n]*(-l1*np.real(F[n])*np.sin(np.angle(dy[n])) - l2_d*(f_d2[n]-f_2[n])/f_2[n])
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + np.exp(1j*np.angle(dy[n])))
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*np.cos(np.angle(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))
        else:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F_d*y[n])
            f_d1[n+1] = f_d1[n] + T*f_d1[n]*(-l1*np.real(F_d*y[n])*np.sin(np.angle(dz[n])) - l2_d*(f_d1[n]-f_1[n])/f_1[n])
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + np.exp(1j*np.angle(dz[n])))
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*np.cos(np.angle(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F_d*z[n])
            f_d2[n+1] = f_d2[n] + T*f_d2[n]*(-l1*np.real(F_d*z[n])*np.sin(np.angle(dy[n])) - l2_d*(f_d2[n]-f_2[n])/f_2[n])
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + np.exp(1j*np.angle(dy[n])))
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*np.cos(np.angle(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))
            #dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + y[n])
            #f_d1[n+1] = f_d1[n] + T*f_d1[n]*(-l1*np.real(y[n])*np.sin(np.angle(dz[n])) - l2_d*(f_d1[n]-f_1[n])/f_1[n])
            #z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + np.exp(1j*np.angle(dz[n])))
            #f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*np.cos(np.angle(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            #dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + z[n])
            #f_d2[n+1] = f_d2[n] + T*f_d2[n]*(-l1*np.real(z[n])*np.sin(np.angle(dy[n])) - l2_d*(f_d2[n]-f_2[n])/f_2[n])
            #y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + np.exp(1j*np.angle(dy[n])))
            #f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*np.cos(np.angle(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))

    #print(f_1_0, f_2_0)
    #plt.plot(np.real(y))
    #plt.plot(f_1)
    #plt.plot(np.real(z))
    #plt.plot(f_2)
    #plt.show()
    
    zpeaks, _ = find_peaks(np.real(z[tr_samps:]))
    ypeaks, _ = find_peaks(np.real(y[tr_samps:]))

    if np.abs(len(zpeaks) - len(ypeaks)) > 2:
        raise ValueError('Different number of peaks between Models')
	
    lag_mat = np.zeros((64,10))
    for j in range(10):
        lag_mat[:,j]=np.abs(zpeaks[4:4+64]+np.random.normal(0, 12, 64) - ypeaks[j:j+64]+np.random.normal(0, 12, 64))
    corr_idx = np.argmin(np.mean(lag_mat,0))
    peaksdiff = np.reshape(lag_mat[:64,corr_idx],(16,4))
    mean_SPR_miss_pairs[i,:] = np.mean(peaksdiff,0)/fs
    print(mean_SPR_miss_pairs)

                  

f_1 = (1000/400)*np.ones(t.shape);  # Adaptive frequency osc1
f_d1 = (1000/400)*np.ones(t.shape);  # Adaptive frequency osc1
f_2 = (1000/400)*np.ones(t.shape);  # Adaptive frequency osc2
f_d2 = (1000/400)*np.ones(t.shape);  # Adaptive frequency osc2

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_match1[i]
    f_d1[0] = 1000/freqs_match1[i]
    f_2[0] = 1000/freqs_match2[i]
    f_d2[0] = 1000/freqs_match2[i]
    f_1_0 = 1000/freqs_match1[i]
    f_2_0 = 1000/freqs_match2[i] 
        
    # Forward Euler integration
    # Forward Euler integration
    for n in range(ntime-1):
        if t[n] <= tr_time:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F[n])
            f_d1[n+1] = f_d1[n] + T*f_d1[n]*(-l1*np.real(F[n])*np.sin(np.angle(dz[n])) - l2_d*(f_d1[n]-f_1[n])/f_1[n])
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + np.exp(1j*np.angle(dz[n])))
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*np.cos(np.angle(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F[n])
            f_d2[n+1] = f_d2[n] + T*f_d2[n]*(-l1*np.real(F[n])*np.sin(np.angle(dy[n])) - l2_d*(f_d2[n]-f_2[n])/f_2[n])
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + np.exp(1j*np.angle(dy[n])))
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*np.cos(np.angle(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))
        else:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F_d*y[n])
            f_d1[n+1] = f_d1[n] + T*f_d1[n]*(-l1*np.real(F_d*y[n])*np.sin(np.angle(dz[n])) - l2_d*(f_d1[n]-f_1[n])/f_1[n])
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + np.exp(1j*np.angle(dz[n])))
            f_1[n+1] = f_1[n] + T*f_1[n]*(-l1*np.cos(np.angle(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,(f_1[n]-f_1_0)/f_1_0)-1))

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F_d*z[n])
            f_d2[n+1] = f_d2[n] + T*f_d2[n]*(-l1*np.real(F_d*z[n])*np.sin(np.angle(dy[n])) - l2_d*(f_d2[n]-f_2[n])/f_2[n])
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + np.exp(1j*np.angle(dy[n])))
            f_2[n+1] = f_2[n] + T*f_2[n]*(-l1*np.cos(np.angle(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,(f_2[n]-f_2_0)/f_2_0)-1))

    zpeaks, _ = find_peaks(np.real(z[tr_samps:]))
    ypeaks, _ = find_peaks(np.real(y[tr_samps:]))

    if np.abs(len(zpeaks) - len(ypeaks)) > 3:
        raise ValueError('Different number of peaks between Models')

    lag_mat = np.zeros((64,10))
    for j in range(10):
        lag_mat[:,j]=np.abs(zpeaks[4:4+64]+np.random.normal(0, 12, 64) - ypeaks[j:j+64]+np.random.normal(0, 12, 64))
    corr_idx = np.argmin(np.mean(lag_mat,0))
    peaksdiff = np.reshape(lag_mat[:64,corr_idx],(16,4))
    mean_SPR_match_pairs[i,:] = np.mean(peaksdiff,0)/fs
    print(mean_SPR_match_pairs)

# Create coupled bar plots
fig, ax = plt.subplots()
ind = np.arange(len(np.mean(mean_SPR_match_pairs, 0))) # x locations for the groups
width = 0.25  # width of the bars
p1 = ax.bar(ind, 1000 * np.mean(mean_SPR_match_pairs, 0), width, bottom=0)
p2 = ax.bar(ind + width, 1000 * np.mean(mean_SPR_miss_pairs, 0), width, bottom=0)

ax.set_title('Mean Absolute Asynchrony (ms)')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1', '2', '3', '4'))

ax.legend((p1[0], p2[0]), ('Match', 'Missmatch'))
ax.autoscale_view()

plt.show()
