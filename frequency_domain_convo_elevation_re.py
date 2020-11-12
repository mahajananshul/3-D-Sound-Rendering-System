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
import scipy.fftpack
import winsound

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
sample_rate, data = wavfile.read(r'E:\d\1st Sem - uOttawa\DSP\Project\Input audio and mat files\helicopter.wav')

## Loading HRTF data file
read_mat = loadmat(r'E:\d\1st Sem - uOttawa\DSP\Project\Input audio and mat files\large_pinna_frontal.mat')

left = read_mat.get('left') #storing left array of data
right = read_mat.get('right') #storing right array of data

## Left side Process
k = 0;
j = 8909;
left_out = np.zeros(882200,np.int16)
for i in range (0,99,1):
    
    ## HRTF data conversion into frequency domain
    left_data=left[:,i] #taking data from a column 
    pad_left=np.zeros(8908,np.int16) #zeros to make it of lenght: n+p-1
    left_data=np.append(left_data,pad_left) #appending zeros in the end
    left_data_fft=scipy.fftpack.fft(left_data) #fast fourier transfer
#    print(f'lenght of fft of data{len(left_data_fft)}')
    left_data_fft_shift=np.fft.ifftshift(left_data_fft) #swaping to get the desired frequency response
#    plt.plot(left_data_fft_shift)
#    plt.show()
    
    ## Audio data conversion into frequency domain
    data_test = data[k:j] #taking some portion of audio file
    pad_data=np.zeros(199,np.int16) #zeros to make it of lenght: n+p-1
    data_test=np.append(data_test,pad_data) #appending zeros in the end
    data_test_fft=scipy.fftpack.fft(data_test) #fast fourier transfer
#    print(f'lenght of fft of audio{len(data_test_fft)}')
    data_test_fft_shift=np.fft.ifftshift(data_test_fft) #swaping to get the desired frequency response
#    plt.plot(data_test_fft_shift)
#    plt.show()
    
    
    freq_left_out=left_data_fft_shift*data_test_fft_shift # multipling in frequency domain
#    print(f'lenght in frequency domain{len(freq_left_out)}')
    freq_left_out=np.fft.ifftshift(freq_left_out)
    
    con_out=scipy.fftpack.ifft(freq_left_out) #convolved data converted in time domain
#    print(f'lenght in time domain{len(con_out)}')
    
    ## Overlap and Add
    x1 = np.append(np.zeros(k,np.int16),con_out) #generating zeros till the current data starts and appending it with current convolved data
    y1 = np.append(x1,np.zeros(873091-k,np.int16)) #appending zeros for remaining part
    left_out = left_out[:882199]
    left_out = left_out + y1 #adding the overlapped data
    k = k+8909;
    j = j+8909;


## Right side Process   
k = 0;
j = 8909;

right_out = np.zeros(882200,np.int16)
for i in range (0,99,1):
    
    ## HRTF data conversion into frequency domain
    right_data=right[:,i]
    pad_right=np.zeros(8908,np.int16) #zeros to make it of lenght: n+p-1
    right_data=np.append(right_data,pad_right) #appending zeros in the end
    right_data_fft=scipy.fftpack.fft(right_data) #fast fourier transfer
#    print(f'lenght of fft of data{len(right_data_fft)}')
    right_data_fft_shift=np.fft.ifftshift(right_data_fft) #swaping to get the desired frequency response
    
    ## Audio data conversion into frequency domain
    data_test = data[k:j]
    pad_data=np.zeros(199,np.int16) #zeros to make it of lenght: n+p-1
    data_test=np.append(data_test,pad_data) #appending zeros in the end
    data_test_fft=scipy.fftpack.fft(data_test) #fast fourier transfer
#    print(f'lenght of fft of audio{len(data_test_fft)}')
    data_test_fft_shift=np.fft.ifftshift(data_test_fft) #swaping to get the desired frequency response
    
    
    freq_right_out=right_data_fft_shift*data_test_fft_shift # multipling in frequency domain
#    print(f'lenght in frequency domain{len(freq_right_out)}')
    freq_right_out=np.fft.ifftshift(freq_right_out)
    
    con_out=scipy.fftpack.ifft(freq_right_out) #convolved data converted in time domain
#    print(f'lenght in time domain{len(con_out)}')
    
    ## Overlap and Add
    x2 = np.append(np.zeros(k,np.int16),con_out) #generating zeros till the current data starts and appending it with current convolved data
    y2 = np.append(x2,np.zeros(873091-k,np.int16))#appending zeros for remaining part
    right_out = right_out[:882199]
    right_out = right_out + y2 #adding the overlapped data
    k = k+8909;
    j = j+8909;

##Parameters for Low-pass filter
order = 6
fs = sample_rate       #sample rate (Hz)
cutoff = 17000  #desired cutoff frequency of the filter (Hz)

# Get the filter coefficients so we can check its frequency response
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.show()


## Use of the filter
T = 20         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)

# Filter the data
data1 = left_out;
data2 = right_out;
left_out = butter_lowpass_filter(data1, cutoff, fs, order) #filtering the left side convolved output
right_out = butter_lowpass_filter(data2, cutoff, fs, order) #filtering the right side convolved output

 
out=np.stack((right_out,left_out),axis=1) #stacking the both side convolved outputs. Here axis =1 means it stacks vertically 
out = np.int16(out/np.max(np.abs(out)) * 32767)

print(f'\nSuccessfully Executed...!')

io.wavfile.write('frequency_domain_elevation.wav', sample_rate, out)
winsound.PlaySound('frequency_domain_elevation.wav', winsound.SND_ASYNC)
