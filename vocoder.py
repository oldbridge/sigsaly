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
from scipy.signal import sosfilt, cheby1

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
        
    def plot_filtered(self):
        fig, axs = plt.subplots(11, 1, sharex=True)
        t = np.arange(0, len(self.data) / self.rate, 1/ self.rate)
        axs[0].plot(t, self.data)
        for i, signal in enumerate(self.filtered):
            axs[i + 1].plot(t, signal)
        axs[i + 1].set_xlabel('Time [s]')
            
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
        self.filtered = values
                
if __name__ == '__main__':
    rec_time = 5
    a = AudioDevice()
    a.load_wav()
    v = Vocoder(a)
    v.plot_filtered()
    #a.record(rec_time)
    #a.show_timeplot()
    #time.sleep(rec_time)
    #a.show_timeplot()
    #a.bandpass_signal()
    #a.play()
    #a.save_wav()
    
    
        
        