#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from nifs.voltdata import *
from nifs.timedata import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import time

class CTS_Analysis:
    def __init__(self):
        pass

    def plot_volt_ctsfosc(self):
        data = VoltData.retrieve('CTSfosc1', 131030, 1, 1)
        voltdata = data.get_val()
        # clk = data.params['ClockSpeed']
        timedata = numpy.arange(0.0, len(voltdata))  # / clk

        plt.title('CTSfosc1')
        plt.xlabel('t [sec]')
        plt.ylabel('v [V]')
        idx = 10000
        plt.plot(timedata[:idx], voltdata[:idx])
        plt.show()

    def stft(self):
        data = VoltData.retrieve('CTSfosc1', 131030, 1, 1)
        voltdata = data.get_val()
        timedata = TimeData.retrieve('CTSfosc1', 131030, 1, 1).get_val()

        idx_st = 0
        idx_ed = 100000

        y = voltdata[idx_st:idx_ed]
        x = timedata[idx_st:idx_ed]
        nperseg = 1e3
        time_offset = 0.0

        MAXFREQ = 1e0
        N = 1e0 * np.abs(1 / (x[1] - x[2]))
        f, t, Zxx = sig.spectrogram(y, fs=N)#, nperseg=nperseg)#, window='hamming', nperseg=nperseg, mode='complex')
        vmax = 0.8*np.max(Zxx)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.pcolormesh(t * 1e-3 + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax)
        ax2.plot(x, y)
        # plt.contourf(t, f, np.abs(Zxx), 200, norm=LogNorm())# vmax=1e-7)
        ax1.set_ylabel("Frequency [Hz]")
        ax2.set_xlabel("Time [sec]")
        #plt.ylim([0, MAXFREQ])
        #plt.xlim([0.5, 2.5])
        plt.show()


if __name__ == "__main__":
    start = time.time()
    ctsa = CTS_Analysis()
    ctsa.stft()
    t = time.time() - start
    print("erapsed time= %.3f sec" % t)
