# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:49:29 2019

@author: Sharan
"""

import numpy as np
from scipy.io import wavfile
from scipy import io
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy import signal
import winsound #to play sound

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

## Parameters for HRTF
c = 343  #speed of sound = 343 m/sec
a = 0.152/2 #radius of head (in meters)
N = 200 #sample lenght
tau = 0.5*(a/c) #value of tau
 
HR = np.zeros((200,72)) #Inintialzing the array size for Right side
HL = np.zeros((200,72)) #Inintialzing the array size for Leftt side

for t in range(0,72,1): #for 72 diffrent angles
    theta = -np.pi*(1/2-t/72); #theta changes by 1.25 degrees
    alpha = (1+np.sin(theta))*0.5; 
#    tau = (theta+np.sin(theta))*a/c #this tau can be used for better results according to research paper    
    Tr = (1-alpha)*tau;
    Tl = alpha * tau;
    
    for k in range(0,200,1):
        w = (2*np.pi*k*44100)/N;
        HL[k][t] = ((1+(1j*2*(1-alpha)*tau*w))*np.exp(-1*1j*w*Tl))/(1+(1j*w*tau)) #Transfer Function for Left side
        HR[k][t] = ((1+(1j*2*alpha*tau*w))*np.exp(-1*1j*w*Tr))/(1+(1j*w*tau)) #Transfer Function for Right side

## Loading audio file
sample_rate, data = wavfile.read(r'E:\d\1st Sem - uOttawa\DSP\Project\Input audio and mat files\helicopter.wav')

left = HL 
right = HR

## Left side Process
k = 0;
j = 12250;
left_out = np.array([],np.int16)
for i in range (0,72,1):
    left_data=left[:,i] #taking data from a column    
    data_test = data[k:j] #taking some portion of audio file 
    k = k+12250;
    j = j+12250;
    con_out = signal.convolve(data_test,left_data) #convolution
    con_out = con_out[99:12349] #taking convolved data of size 12250
    left_out = np.append(left_out,con_out) #appending the current convolved data with recent

## Right side Process    
k = 0;
j = 12250;
right_out = np.array([],np.int16)
for i in range (0,72,1):
    right_data=right[:,i] #taking data from a column 
    data_test = data[k:j] #taking some portion of audio file
    k = k+12250;
    j = j+12250;
    con_out = signal.convolve(data_test,right_data) #convolution
    con_out = con_out[99:12349] #taking convolved data of size 12250
    right_out = np.append(right_out,con_out) #appending the current convolved data with recent
 
# Filter requirements.
order = 6
fs = sample_rate  # sample rate, Hz
cutoff = 18000  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

## Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


## Demonstrate the use of the filter.
# First make some data to be filtered.
T = 20         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)


## Filter the data, and plot both the original and filtered signals.
data1 = left_out;
left_out = butter_lowpass_filter(data1, cutoff, fs, order) #filtering the left side convolved output

data2 = right_out;
right_out = butter_lowpass_filter(data2, cutoff, fs, order) #filtering the right side convolved output

plt.subplot(2,1,2)
plt.plot(t, data1, 'b-', label='data')
plt.plot(t, left_out, 'g-', linewidth=2, label='filtered data')
plt.title("Left Side")
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
plt.subplots_adjust(hspace=0.7)
plt.show()

    
out=np.stack((left_out,right_out),axis=1) #stacking the both side convolved outputs. Here axis =1 means it stacks vertically
out = np.int16(out/np.max(np.abs(out)) * 65525)
print(f'\nSuccessfully Executed...!')

io.wavfile.write('time_hrtf.wav', sample_rate, out)
winsound.PlaySound('time_hrtf.wav', winsound.SND_ASYNC)
