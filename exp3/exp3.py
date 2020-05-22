import numpy as np 
import matplotlib.pyplot as plt

from math import pi

# Simulation parameters
fs        = 2000
T         = 1/fs
dur       = 52
t         = np.linspace(0, dur, dur*fs)
ntime     = t.size
halfsamps = np.floor(ntime/2);


# z - parameters
a_d = -1
b_d = 4
b2_d = -0.25
F_d = 1.5
a = 1
b = -1
#%%%%%%%%%%%%%%%%%% HEBBIAN LEARNING PARAMETERS %%%%%%%%%%%%%%
l1 = 0.75 # learning rate
l2 = 0.04 # 0.00019 # elasticity
l2_d = l2 # 0.00019 # elasticity
base=2

z = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator1
dz = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size)
y = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator2
dy = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size) 

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
tr_samps = tr_time * 1000  # training samples

locs_z = pks_z = locs_F = pks_F = locs_y = pks_y = [] # Locations @ local maxima

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_miss1[i]
    f_d1[0] = 1000/freqs_miss1[i]
    f_2[0] = 1000/freqs_miss2[i] 
    f_d2[0] = 1000/freqs_miss2[i] 
        
    # Forward Euler integration
    for n in range(ntime-1):
        if t[n] <= tr_time:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F_d*F[n])
            f_d1[n+1] = f_d1[n] + T*(-f_d1[n]*l1*np.real(F_d*F[n])*np.sin(np.angle(dz[n])) - l2_d*(np.power(base,f_d1[n])-np.power(base,f_1[n]))/base)
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + dz[n])
            f_1[n+1] = f_1[n] + T*(-f_1[n]*l1*(np.real(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,f_1[n])-np.power(base,f_1[0]))/base)

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F_d*F[n])
            f_d2[n+1] = f_d2[n] + T*(-f_d2[n]*l1*np.real(F_d*F[n])*np.sin(np.angle(dy[n])) - l2_d*(np.power(base,f_d2[n])-np.power(base,f_2[n]))/base)
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + dy[n])
            f_2[n+1] = f_2[n] + T*(-f_2[n]*l1*(np.real(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,f_2[n])-np.power(base,f_2[0]))/base)
        else:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F_d*y[n])
            f_d1[n+1] = f_d1[n] + T*(-f_d1[n]*l1*np.real(F_d*y[n])*np.sin(np.angle(dz[n])) - l2_d*(np.power(base,f_d1[n])-np.power(base,f_1[n]))/base)
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + dz[n])
            f_1[n+1] = f_1[n] + T*(-f_1[n]*l1*(np.real(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,f_1[n])-np.power(base,f_1[0]))/base)

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F_d*z[n])
            f_d2[n+1] = f_d2[n] + T*(-f_d2[n]*l1*np.real(F_d*z[n])*np.sin(np.angle(dy[n])) - l2_d*(np.power(base,f_d2[n])-np.power(base,f_2[n]))/base)
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + dy[n])
            f_2[n+1] = f_2[n] + T*(-f_2[n]*l1*(np.real(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,f_2[n])-np.power(base,f_2[0]))/base)

        # Find local maxima and location - (zero crossing)
        if (z[n+1].imag >= 0.0) and (z[n].imag <= 0.0):
            locs_z  = np.append(locs_z, n+1)
            pks_z   = np.append(pks_z, z[n+1].real)
        if (y[n+1].imag >= 0.0) and (y[n].imag <= 0.0):
            locs_y  = np.append(locs_y, n+1)
            pks_y   = np.append(pks_y, y[n+1].real)

    f_1trsamps  = np.take(f_1, tr_samps)
    f_2trsamps  = np.take(f_2, tr_samps)

    # Finding leader
    find_leader = []
    find_leader = np.append(find_leader, np.absolute(f_s - f_1trsamps.real))
    find_leader = np.append(find_leader, np.absolute(f_s - f_2trsamps.real))
    # Get leader
    leader = np.minimum(np.array(np.absolute(f_s - f_2trsamps.real)), np.array(np.absolute(f_s - f_1trsamps.real)))

    # Find which oscillator is more similar to stimulus
    which_min = np.where(find_leader == leader)

    if which_min[0][0] == 0:
        locs_lead = locs_z
        locs_follow = locs_y
    else:
        locs_lead = locs_y
        locs_follow = locs_z      

    new_followlocs = np.zeros(len(locs_lead));

    for iloc in range(0, len(locs_lead)):
        locsy_diff = np.absolute(locs_follow - locs_lead[iloc])
        nypeak_index = np.argmin(locsy_diff) # get index of minimum
        new_followlocs[iloc] = locs_follow[nypeak_index]
                          
    # find the index after training  
    tr_samps_locsz_diff = np.absolute(tr_samps - locs_lead)
    nzpeak_index = np.argmin(tr_samps_locsz_diff)
    mid_nzpeak = locs_lead[nzpeak_index]
    # eliminate the training part of the simulation for z
    locs_lead = locs_lead[nzpeak_index:]
    # eliminate the training part of the simulation for y
    new_followlocs = new_followlocs[nzpeak_index:]               
    # calculate number of peaks divisible by 4
    mod_four1 = np.mod(locs_lead.size, 4)
    # eliminate extra peaks
    locs_lead = locs_lead[0:(locs_lead.size)-mod_four1]
    new_followlocs = new_followlocs[0:(new_followlocs.size)-mod_four1]

    # Recover Variable names for computation
    if which_min[0][0] == 0:
        locs_z = locs_lead
        locs_y = new_followlocs
    else:
        locs_y = locs_lead
        locs_z = new_followlocs

    # Mean Asynchrony - break locations vector into 4
    # Note: 4 = cycles/repetitions of synchronization
    z_locsFourCycles = locs_z.reshape((-1, 4), order='F') # quemamada "F"
    y_locsFourCycles = locs_y.reshape((-1, 4), order='F')
    
    z_locsFourCycles = z_locsFourCycles.astype(int)
    
    y_locsFourCycles = y_locsFourCycles.astype(int)

    # Take mean of asynchronies over the four repetitions.
    mean_SPR_miss_pairs[i,:] = np.mean(np.absolute(np.take(t, z_locsFourCycles) - np.take(t, y_locsFourCycles)), 0)
    locs_z = pks_z = locs_F = pks_F = locs_y = pks_y = [] # Refresh ocations @ local maxima
                  

f_1 = np.zeros(t.shape);  # Adaptive frequency osc1
f_d1 = np.zeros(t.shape);  # Adaptive frequency osc1
f_2 = np.zeros(t.shape);  # Adaptive frequency osc2
f_d2 = np.zeros(t.shape);  # Adaptive frequency osc2

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_match1[i]
    f_d1[0] = 1000/freqs_match1[i]
    f_2[0] = 1000/freqs_match2[i] 
    f_d2[0] = 1000/freqs_match2[i] 
        
    # Forward Euler integration
    # Forward Euler integration
    for n in range(ntime-1):
        if t[n] <= tr_time:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F_d*F[n])
            f_d1[n+1] = f_d1[n] + T*(-f_d1[n]*l1*np.real(F_d*F[n])*np.sin(np.angle(dz[n])) - l2_d*(np.power(base,f_d1[n])-np.power(base,f_1[n]))/base)
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + dz[n])
            f_1[n+1] = f_1[n] + T*(-f_1[n]*l1*(np.real(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,f_1[n])-np.power(base,f_1[0]))/base)

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F_d*F[n])
            f_d2[n+1] = f_d2[n] + T*(-f_d2[n]*l1*np.real(F_d*F[n])*np.sin(np.angle(dy[n])) - l2_d*(np.power(base,f_d2[n])-np.power(base,f_2[n]))/base)
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + dy[n])
            f_2[n+1] = f_2[n] + T*(-f_2[n]*l1*(np.real(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,f_2[n])-np.power(base,f_2[0]))/base)
        else:
            dz[n+1] = dz[n] + T*f_d1[n]*(dz[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dz[n]),2)) + b2_d*np.power(np.abs(dz[n]),4)/(1-np.power(np.abs(dz[n]),2))) + F_d*y[n])
            f_d1[n+1] = f_d1[n] + T*(-f_d1[n]*l1*np.real(F_d*y[n])*np.sin(np.angle(dz[n])) - l2_d*(np.power(base,f_d1[n])-np.power(base,f_1[n]))/base)
            z[n+1] = z[n] + T*f_1[n]*(z[n]*(a + 1j*2*pi + b*(abs(z[n])**2)) + dz[n])
            f_1[n+1] = f_1[n] + T*(-f_1[n]*l1*(np.real(dz[n]))*np.sin(np.angle(z[n])) - l2*(np.power(base,f_1[n])-np.power(base,f_1[0]))/base)

            dy[n+1] = dy[n] + T*f_d2[n]*(dy[n]*(a_d + 1j*2*np.pi + b_d*(np.power(np.abs(dy[n]),2)) + b2_d*np.power(np.abs(dy[n]),4)/(1-np.power(np.abs(dy[n]),2))) + F_d*z[n])
            f_d2[n+1] = f_d2[n] + T*(-f_d2[n]*l1*np.real(F_d*z[n])*np.sin(np.angle(dy[n])) - l2_d*(np.power(base,f_d2[n])-np.power(base,f_2[n]))/base)
            y[n+1] = y[n] + T*f_2[n]*(y[n]*(a + 1j*2*pi + b*(abs(y[n])**2)) + dy[n])
            f_2[n+1] = f_2[n] + T*(-f_2[n]*l1*(np.real(dy[n]))*np.sin(np.angle(y[n])) - l2*(np.power(base,f_2[n])-np.power(base,f_2[0]))/base)

        # Find local maxima and location - (zero crossing)
        if (z[n+1].imag >= 0.0) and (z[n].imag <= 0.0):
            locs_z  = np.append(locs_z, n+1)
            pks_z   = np.append(pks_z, z[n+1].real)
        if (y[n+1].imag >= 0.0) and (y[n].imag <= 0.0):
            locs_y  = np.append(locs_y, n+1)
            pks_y   = np.append(pks_y, y[n+1].real)

    f_1trsamps  = np.take(f_1, tr_samps)
    f_2trsamps  = np.take(f_2, tr_samps)

    # Finding leader
    find_leader = []
    find_leader = np.append(find_leader, np.absolute(f_s - f_1trsamps.real))
    find_leader = np.append(find_leader, np.absolute(f_s - f_2trsamps.real))
    # Get leader
    leader = np.minimum(np.array(np.absolute(f_s - f_2trsamps.real)), np.array(np.absolute(f_s - f_1trsamps.real)))

    # Find which oscillator is more similar to stimulus
    which_min = np.where(find_leader == leader)

    if which_min[0][0] == 0:
        locs_lead = locs_z
        locs_follow = locs_y
    else:
        locs_lead = locs_y
        locs_follow = locs_z      

    new_followlocs = np.zeros(len(locs_lead));

    for iloc in range(0, len(locs_lead)):
        locsy_diff = np.absolute(locs_follow - locs_lead[iloc])
        nypeak_index = np.argmin(locsy_diff) # get index of minimum
        new_followlocs[iloc] = locs_follow[nypeak_index]
                          
    # find the index after training  
    tr_samps_locsz_diff = np.absolute(tr_samps - locs_lead)
    nzpeak_index = np.argmin(tr_samps_locsz_diff)
    mid_nzpeak = locs_lead[nzpeak_index]
    # eliminate the training part of the simulation for z
    locs_lead = locs_lead[nzpeak_index:]
    # eliminate the training part of the simulation for y
    new_followlocs = new_followlocs[nzpeak_index:]               
    # calculate number of peaks divisible by 4
    mod_four1 = np.mod(locs_lead.size, 4)
    # eliminate extra peaks
    locs_lead = locs_lead[0:(locs_lead.size)-mod_four1]
    new_followlocs = new_followlocs[0:(new_followlocs.size)-mod_four1]

    # Recover Variable names for computation
    if which_min[0][0] == 0:
        locs_z = locs_lead
        locs_y = new_followlocs
    else:
        locs_y = locs_lead
        locs_z = new_followlocs

    # Mean Asynchrony - break locations vector into 4
    # Note: 4 = cycles/repetitions of synchronization
    z_locsFourCycles = locs_z.reshape((-1, 4), order='F') # quemamada "F"
    y_locsFourCycles = locs_y.reshape((-1, 4), order='F')
    
    z_locsFourCycles = z_locsFourCycles.astype(int)
    
    y_locsFourCycles = y_locsFourCycles.astype(int)

    # Take mean of asynchronies over the four repetitions.
    mean_SPR_match_pairs[i,:] = np.mean(np.absolute(np.take(t, z_locsFourCycles) - np.take(t, y_locsFourCycles)), 0)
    locs_z = pks_z = locs_F = pks_F = locs_y = pks_y = [] # Refresh ocations @ local maxima


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
