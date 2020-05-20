# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:49:29 2019

@author: Sharan
"""

import numpy as np
from scipy.io import wavfile
from scipy.io import loadmat
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

## Loading audio file
sample_rate, data = wavfile.read(r'C:\Users\Sharan\Desktop\DSP\Project\Final\helicopter.wav')

## Loading HRTF data file
read_mat = loadmat(r'C:\Users\Sharan\Desktop\DSP\Project\Final\large_pinna_final.mat')
print(f'Lenght of data file = {len(read_mat)}')

left = read_mat.get('left') #storing left array of data
right = read_mat.get('right') #storing right array of data

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
cutoff = 7000  # desired cutoff frequency of the filter, Hz

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

    
out=np.stack((right_out,left_out),axis=1) #stacking the both side convolved outputs. Here axis =1 means it stacks vertically
out = np.int16(out/np.max(np.abs(out)) * 32767)
print(f'\nSuccessfully Executed...!')

io.wavfile.write('time_domain_azimuth.wav', sample_rate, out)
winsound.PlaySound('time_domain_azimuth.wav', winsound.SND_ASYNC)
