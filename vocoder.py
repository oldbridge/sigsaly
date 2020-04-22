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
    
    def show_timeplot(self):
        plt.plot(np.linspace(0, len(self.frames) / self.rate, len(self.frames)), self.frames)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [normalized]")
        plt.grid(True)
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
        
if __name__ == '__main__':
    rec_time = 3
    a = AudioDevice()
    a.load_wav()
    #a.record(rec_time)
    a.show_timeplot()
    #time.sleep(rec_time)
    #a.show_timeplot()
    a.play()
    #a.save_wav()
    
    
        
        