#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:33:52 2020

@author: oldbridge
"""

import pyaudio
import wave
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import sosfilt, cheby1, correlate

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def show_timeplot(frames, rate):
    plt.plot(np.linspace(0, len(frames) / rate, len(frames)), frames)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [normalized]")
    plt.grid(True)
        
class AudioDevice():
    def __init__(self):
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.filename = "file.wav"
        self.audio = pyaudio.PyAudio()
        
    def record(self, rec_time=3):
        self.frames = []
        stream = self.audio.open(format=self.format, 
                            channels=self.channels,
                            rate=self.rate, input=True,
                            frames_per_buffer=self.chunk)
        
        for i in range(0, int(self.rate / self.chunk * rec_time)):
            data = stream.read(self.chunk)
            data = np.frombuffer(data, dtype=np.int16)
            self.frames.extend(data)
        # stop Recording
        stream.stop_stream()
        stream.close()
        self.frames = np.array(self.frames)
        self.frames = self.frames / 32768 # 16 bit int from -32768 to 32768
    
    def __to_play_format(self, array):
        return (array * 32768).astype(np.int16).tostring()
        
    def play(self):
        stream = self.audio.open(format=self.format,
                                 rate=self.rate,
                                 channels=self.channels,
                                 output=True)
        
        stream.write(self.__to_play_format(self.frames))
    
    def save_wav(self, filename='audio.wav'):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join((self.frames * 32768).astype(np.int16)))
        wf.close()

    def load_wav(self, filename='audio.wav'):
        # Open the sound file 
        self.rate, data = wavfile.read(filename)
        # Play the sound by writing the audio data to the stream
        self.frames = data / 32768
    
    def bandpass_signal(self):
        self.frames = butter_bandpass_filter(self.frames, 
                                             150, 2950, self.rate)
    
class Vocoder():
    def __init__(self, audio_device):
        self.ad = audio_device
        self.data = self.ad.frames
        self.rate = self.ad.rate
        self.filter_bank()
        self.pitch_detector_autocorr()
        
    def plot_filtered(self):
        fig, axs = plt.subplots(11, 1, sharex=True)
        t = np.arange(0, len(self.data) / self.rate, 1/ self.rate)
        axs[0].plot(t, self.data)
        for i, signal in enumerate(self.filtered):
            axs[i + 1].plot(t, signal)
        axs[i + 1].set_xlabel('Time [s]')
    
    def plot_filtered_sampled(self):
        fig, axs = plt.subplots(11, 1, sharex=True)
        t = np.arange(0, self.sampled_weights.shape[1] * 20e-3, 20e-3)
        for i, signal in enumerate(self.sampled_weights):
            axs[i].plot(t, signal)
        axs[i].set_xlabel('Time [s]')
            
    def filter_bank(self):
        values = np.zeros((10, len(self.data)))
        filter_bank = [[150, 350],
                       [350, 550],
                       [550, 750],
                       [750, 1050],
                       [1050, 1450],
                       [1450, 1750],
                       [1750, 2000],
                       [2000, 2350],
                       [2350, 2650],
                       [2650, 2950]]
        
        # Low pass filter of 25 Hz
        sos_lp = cheby1(10, 1, 25, 'lp', fs=self.rate, output='sos')
        for i, band in enumerate(filter_bank):
            print(f"Filtering from {band[0]} to {band[1]}")
            # First apply bandpass
            sos_bp = cheby1(10, 1, band, 'bandpass', fs=self.rate, output='sos')
            filtered = sosfilt(sos_bp, self.data)
            
            # Rectify
            filtered[filtered < 0] = 0
            
            # Apply 25 Hz lowpass filter
            filtered = sosfilt(sos_lp, filtered)
            values[i, :] = filtered
        self.weights = values
    
    def pitch_detector_autocorr(self):
        win_size = 400
        self.pitch_rate = self.rate / win_size
        print(f"Pitch detection frequency: {self.pitch_rate} Hz")
        freqs = []
        
        # First of all filter signal to interested band
        sos_bp = cheby1(10, 1, [150, 2950], 'bandpass', fs=self.rate, output='sos')
        data_mod = sosfilt(sos_bp, self.data)
        
        for i in range(0, len(data_mod) - win_size, win_size):
            # Calculate autocorrelation and throw away the negative lags
            sig = data_mod[i:i+win_size]
            corr = correlate(sig, sig, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find the first low point
            d = np.diff(corr)
            start = np.nonzero(d > 0)[0][0]
            # Find the next peak after the low point (other than 0 lag).  This bit is
            # not reliable for long signals, due to the desired peak occurring between
            # samples, and other peaks appearing higher.
            # Should use a weighting function to de-emphasize the peaks at longer lags.
            peak = np.argmax(corr[start:]) + start
            px = corr[peak]

            freqs.append(self.rate / px)
        
        # Apply 25 Hz LP filter
        sos_lp = cheby1(10, 1, 25, 'lp', fs=self.pitch_rate, output='sos')
        freqs_lp = sosfilt(sos_lp, np.array(freqs))
        self.pitches = freqs_lp
    
    def sample(self):
        rate_s = 20e-3  # One sample every 20ms
        
        # First sample the weights
        self.sampled_weights = self.weights[:, 0::int(self.rate * rate_s)]
        
        # Then sample the frequencies
        self.sampled_pitches = self.pitches[0::int(self.pitch_rate * rate_s)]
        #self.sampled_pitches = self.sampled_pitches[]
        
        print(len(self.sampled_pitches))
    def synthesize(self):
        pass
        
if __name__ == '__main__':
    rec_time = 5
    a = AudioDevice()
    a.load_wav()
    v = Vocoder(a)
    v.sample()
    v.plot_filtered_sampled()
    #a.record(rec_time)
    #a.show_timeplot()
    #time.sleep(rec_time)
    #a.show_timeplot()
    #a.bandpass_signal()
    #a.play()
    #a.save_wav()
    
    
        
        