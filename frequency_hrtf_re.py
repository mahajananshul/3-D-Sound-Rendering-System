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
import scipy.fftpack
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
sample_rate, data = wavfile.read(r'C:\Users\Sharan\Desktop\DSP\Project\Final\helicopter.wav')

left = HL 
right = HR

## Left side Process
k = 0;
j = 12250;
left_out = np.zeros(882200,np.int16)
for i in range (0,72,1):
    
    ## HRTF data conversion into frequency domain
    left_data=left[:,i] #taking data from a column 
    pad_left=np.zeros(12249,np.int16) #zeros to make it of lenght: n+p-1
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
    y1 = np.append(x1,np.zeros(869750-k,np.int16)) #appending zeros for remaining part
    left_out = left_out[:882199]
    left_out = left_out + y1 #adding the overlapped data
    k = k+12250;
    j = j+12250;


## Right side Process   
k = 0;
j = 12250;

right_out = np.zeros(882200,np.int16)
for i in range (0,72,1):
    
    ## HRTF data conversion into frequency domain
    right_data=right[:,i]
    pad_right=np.zeros(12249,np.int16) #zeros to make it of lenght: n+p-1
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
    y2 = np.append(x2,np.zeros(869750-k,np.int16))#appending zeros for remaining part
    right_out = right_out[:882199]
    right_out = right_out + y2 #adding the overlapped data
    k = k+12250;
    j = j+12250;

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

 
out=np.stack((left_out,right_out),axis=1) #stacking the both side convolved outputs. Here axis =1 means it stacks vertically 
out = np.int16(out/np.max(np.abs(out)) * 65525)

print(f'\nSuccessfully Executed...!')
#print(out)

io.wavfile.write('frequency_hrtf.wav', sample_rate, out)
winsound.PlaySound('frequency_hrtf.wav', winsound.SND_ASYNC) #to play sound write after execution