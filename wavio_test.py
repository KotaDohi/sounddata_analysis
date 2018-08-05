#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:53:34 2017

@author: Dohi
"""

import wavio
import numpy as np
import scipy.io.wavfile as siw
import matplotlib.pyplot as plt
"""
filename = "normal_1800rpm_2"

wav = wavio.read(filename+".wav")

x = wav.data
rate = 44100


wavio.write("audio/normal_1800rpm_2.wav",x,wav.rate,sampwidth=3)
"""
import wavio


class MyClass:
    def _init_(self):
        self.filename = ""
        
    def cal(self,filename):
        wav = wavio.read(filename+".wav")
        x = wav.data
        wavio.write("audio/normal_1900rpm_2.wav",x,wav.rate,sampwidth=3)
        
a1 = MyClass()
a1.filename = "normal_1800rpm_2"
a1.cal(filename)