#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from nifs.voltdata import *
from nifs.timedata import *
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import time

class CTS_Analysis:
    def __init__(self):
        pass

    def plot_volt_ctsfosc(self):
        data = VoltData.retrieve('CTSfosc1', 125687, 1, 1)
        voltdata = data.get_val()
        # clk = data.params['ClockSpeed']
        timedata = numpy.arange(0.0, len(voltdata))  # / clk

        plt.title('CTSfosc1')
        plt.xlabel('t [sec]')
        plt.ylabel('v [V]')
        idx = 10000
        plt.plot(timedata[:idx], voltdata[:idx])
        plt.show()

    def load_data(self, shotNo, isLocal=False):
        start = time.time()
        print("Start loading SN%d" % int(shotNo))
        if isLocal:
            npz_comp = np.load('data_buf.npz')
            voltdata = npz_comp['arr_0']
            timedata = npz_comp['arr_1']
        else:
            data = VoltData.retrieve('CTSfosc1', shotNo, 1, 1)

            voltdata = data.get_val()
            timedata = TimeData.retrieve('CTSfosc1', shotNo, 1, 1).get_val()
            #np.savez_compressed('data_buf', voltdata, timedata)

        t = time.time() - start
        print("erapsed time for LOAD = %.3f sec" % t)

        return voltdata, timedata

    def interactive_fft(self, shotNo, f, t, Zxx, num_plot=10, y_range=0, isLog=False):
        plt.figure(figsize=(5,10), dpi=150)

        print('start mean stft')
        num_mean = int(len(t)/num_plot)
        for i in range(num_plot):
            legend = 't=%.4fsec' % t[int(num_mean*(i+0.5))]
            plt.subplot(num_plot, 1, i+1)
            plt.plot(f, np.mean(Zxx[:, num_mean*i:num_mean*(i+1)], axis=1), color='blue', label=legend)
            if i == 0:
                plt.title('#%d' % int(shotNo), loc='right')
            if y_range > 0:
                plt.ylim([0, y_range])
            if isLog:
                plt.yscale('log')
            plt.legend()
        #plt.plot(f, np.mean(Zxx[:, :], axis=1), color='blue')
        plt.xlabel("Frequency [GHz]")
        #plt.set_yscale('log')
        #ax1.set_xlim([np.min(x), np.max(x)])
        #ax2.set_xlim([np.min(x), np.max(x)])
        #ax1.set_ylim([np.min(f), np.max(f)])
        print('start show')
        #fig.tight_layout()
        plt.show()
        plt.clf()

    def interactive_stft_raw_plot(self, shotNo, x, y, f, t, Zxx):
        start = time.time()
        cof_vmax = float(input('Enter cof_vmax: '))
        val_freq_st = float(input('Enter val_freq_st: '))
        val_freq_ed = float(input('Enter val_freq_ed: '))
        val_time_st = float(input('Enter val_time_st: '))
        val_time_ed = float(input('Enter val_time_ed: '))
        vmax = cof_vmax*np.max(Zxx)
        idx_freq_st = getNearestIndex(f, val_freq_st)
        idx_freq_ed = getNearestIndex(f, val_freq_ed)
        idx_time_st = getNearestIndex(t, val_time_st)
        idx_time_ed = getNearestIndex(t, val_time_ed)

        ncols = 1
        nrows = 5
        grid = GridSpec(nrows, ncols, left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)
        fig = plt.figure(figsize=(8, 6), dpi=150)
        fig.clf()
        ax1 = fig.add_subplot(grid[0:4, 0])
        plt.title('CTSfosc1 #%d' % shotNo)#, loc='left')
        ax2 = fig.add_subplot(grid[4, 0])
        plt.figure(figsize=(8, 6), dpi=150)
        im0 = ax1.pcolormesh(t[idx_time_st:idx_time_ed], f[idx_freq_st:idx_freq_ed], np.abs(Zxx[idx_freq_st:idx_freq_ed, idx_time_st:idx_time_ed]), vmin=0, vmax=vmax, cmap='inferno')
        ax2.plot(x[::2000], y[::2000])
        ax1.set_ylabel("Frequency [GHz]")
        ax2.set_xlabel("Time [sec]")
        ax2.set_ylabel("CTS [V]")
        # plt.ylim([0, MAXFREQ])
        #ax1.set_xlim([np.min(x), np.max(x)])
        ax2.set_xlim(t[idx_time_st], t[idx_time_ed] + (t[idx_time_ed]-t[idx_time_st])*0.12)
        #plt.colorbar()
        fig.colorbar(im0, ax=ax1, pad=0.01, fraction=0.1)
        plt.show()
        plt.clf()

        t_end = time.time() - start
        print("erapsed time for plotting STFT = %.3f sec" % t_end)

    def interactive_stft_plot(self, shotNo, f, t, Zxx):
        start = time.time()
        cof_vmax = float(input('Enter cof_vmax: '))
        val_freq_st = float(input('Enter val_freq_st: '))
        val_freq_ed = float(input('Enter val_freq_ed: '))
        val_time_st = float(input('Enter val_time_st: '))
        val_time_ed = float(input('Enter val_time_ed: '))
        vmax = cof_vmax*np.max(Zxx)
        idx_freq_st = getNearestIndex(f, val_freq_st)
        idx_freq_ed = getNearestIndex(f, val_freq_ed)
        idx_time_st = getNearestIndex(t, val_time_st)
        idx_time_ed = getNearestIndex(t, val_time_ed)
        plt.figure(figsize=(8, 6), dpi=150)
        plt.pcolormesh(t[idx_time_st:idx_time_ed], f[idx_freq_st:idx_freq_ed], np.abs(Zxx[idx_freq_st:idx_freq_ed, idx_time_st:idx_time_ed]), vmin=0, vmax=vmax, cmap='inferno')
        plt.title('#%d' % shotNo, loc='right')
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [GHz]")
        plt.colorbar()
        plt.show()
        plt.clf()

        t_end = time.time() - start
        print("erapsed time for plotting STFT = %.3f sec" % t_end)

    def interactive_stft(self, shotNo, timedata, voltdata, val_time_show_st, val_time_show_ed, val_time_st, val_time_ed, val_time2_st, val_time2_ed, isPlot, isIndexing):
        plt.clf()
        fig = plt.figure(figsize=(6,8),dpi=150)
        nperseg = int(input('Enter nperseg: '))
        noverlap = int(input('Enter noverlap : '))
        nfft = int(input('Enter nfft : '))
        cof_vmax = float(input('Enter cof_vmax: '))
        if nperseg == 0:
            nperseg = 2**12
        if noverlap == 0:
            noverlap = nperseg // 8
        if nfft == 0:
            nfft = nperseg
        start = time.time()
        if isIndexing:
            print('start idx')
            buf_val_time_show_st = val_time_show_st
            buf_val_time_show_ed = val_time_show_ed
            buf_val_time_st = val_time_st
            buf_val_time_ed = val_time_ed
            buf_val_time2_st = val_time2_st
            buf_val_time2_ed = val_time2_ed
            self.idx_st = getNearestIndex(timedata, val_time_show_st)
            self.idx_ed = getNearestIndex(timedata, val_time_show_ed)
            print('end idx')
        idx_st = self.idx_st
        idx_ed = self.idx_ed
        #idx_st = 0
        #idx_ed = 900000000
        freq_LO = 74e9

        y = voltdata[idx_st:idx_ed]
        x = timedata[idx_st:idx_ed]
        time_offset = x[0]

        MAXFREQ = 1e0
        N = 1e0 * np.abs(1 / (x[1] - x[2]))
        print('start spectgram')
        f, t, Zxx = sig.spectrogram(y, fs=N, nperseg=nperseg, noverlap=noverlap, nfft=nfft)#, window='hamming', nperseg=nperseg, mode='complex')
        f += freq_LO
        f /= 1e9
        print('end spectgram')
        #vmax = 0.05*np.max(Zxx)
        vmax = cof_vmax*np.max(Zxx)
        if isIndexing:
            print('start idx')
            self.idx_time_st = getNearestIndex(t, val_time_st - time_offset)
            self.idx_time_ed = getNearestIndex(t, val_time_ed - time_offset)
            self.idx_time2_st = getNearestIndex(t, val_time2_st - time_offset)
            self.idx_time2_ed = getNearestIndex(t, val_time2_ed - time_offset)
            print('end idx')
        idx_time_st = self.idx_time_st
        idx_time_ed = self.idx_time_ed
        idx_time2_st = self.idx_time2_st
        idx_time2_ed = self.idx_time2_ed


        print('start plot')
        if isPlot:
            ncols = 4
            nrows = 4
            grid = GridSpec(nrows, ncols, left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)
            fig = plt.figure(figsize=(8,6), dpi=150)
            fig.clf()
            ax1 = fig.add_subplot(grid[0:3, 0:3])
            plt.title('#%d' % shotNo, loc='right')
            ax2 = fig.add_subplot(grid[3, 0:3])
            ax3 = fig.add_subplot(grid[0:3, 3])

            #fig = plt.figure()
            #ax1 = fig.add_subplot(311)
            #plt.title('#%d' % shotNo, loc='right')
            #ax2 = fig.add_subplot(312)
            #ax3 = fig.add_subplot(313)
            print('start pcolormesh')
            ax1.pcolormesh(t + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax, cmap='inferno')
            ax1.axvline(val_time_st, ls="-", color="blue")
            ax1.axvline(val_time_ed, ls="-", color="blue")
            ax1.axvline(val_time2_st, ls="-", color="red")
            ax1.axvline(val_time2_ed, ls="-", color="red")
            print('start xyplot')
            ax2.plot(x[::100], y[::100])
            ax2.axvline(val_time_st, ls="-", color="blue")
            ax2.axvline(val_time_ed, ls="-", color="blue")
            ax2.axvline(val_time2_st, ls="-", color="red")
            ax2.axvline(val_time2_ed, ls="-", color="red")
            print('start mean stft')
            ax3.plot(np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1), f, color='blue')
            ax3.plot(np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1), f, color='red')
            ax3.plot(np.mean(Zxx[:, idx_time_st:idx_time_ed])-np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1), f, color='green')
            ax1.set_ylabel("Frequency [GHz]")
            ax2.set_xlabel("Time [sec]")
            ax3.set_xscale('log')
            #plt.ylim([0, MAXFREQ])
            ax1.set_xlim([np.min(x), np.max(x)])
            ax2.set_xlim([np.min(x), np.max(x)])
            ax3.set_ylim([np.min(f), np.max(f)])
            print('start show')
            #fig.tight_layout()
            plt.show()
            plt.clf()
            #plt.savefig('plot.png')
            print('end plot')
        t_end = time.time() - start
        print("erapsed time for STFT = %.3f sec" % t_end)

        t += time_offset
        return x, y, f, t, Zxx

    def stft(self, shotNo, val_time_st, val_time_ed, val_time2_st, val_time2_ed, isLocal=False):
        if isLocal:
            npz_comp = np.load('data_buf.npz')
            voltdata = npz_comp['arr_0']
            timedata = npz_comp['arr_1']
        else:
            data = VoltData.retrieve('CTSfosc1', shotNo, 1, 1)

            #data = VoltData.retrieve('mwscat2', 125687, 1, 0)
            voltdata = data.get_val()
            timedata = TimeData.retrieve('CTSfosc1', shotNo, 1, 1).get_val()
            #timedata = TimeData.retrieve('mwscat2', 125687, 1, 1).get_val()
            #np.savez_compressed('data_buf', voltdata, timedata)


        idx_st = 0
        idx_ed = 500000000 #shape: 1000000000

        y = voltdata[idx_st:idx_ed]
        x = timedata[idx_st:idx_ed]
        nperseg = 2**8
        time_offset = x[0]

        MAXFREQ = 1e0
        N = 1e0 * np.abs(1 / (x[1] - x[2]))
        f, t, Zxx = sig.spectrogram(y, fs=N)#, nperseg=nperseg)#, window='hamming', nperseg=nperseg, mode='complex')
        vmax = 0.05*np.max(Zxx)

        idx_time_st = getNearestIndex(t, val_time_st - time_offset)
        idx_time_ed = getNearestIndex(t, val_time_ed - time_offset)
        idx_time2_st = getNearestIndex(t, val_time2_st - time_offset)
        idx_time2_ed = getNearestIndex(t, val_time2_ed - time_offset)

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        plt.title('#%d' % shotNo, loc='right')
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #ax1.pcolormesh(t * 1e-3 + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax)
        ax1.pcolormesh(t + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax)
        ax1.axvline(val_time_st, ls="-", color="blue")
        ax1.axvline(val_time_ed, ls="-", color="blue")
        ax1.axvline(val_time2_st, ls="-", color="red")
        ax1.axvline(val_time2_ed, ls="-", color="red")
        ax2.plot(x, y)
        ax2.axvline(val_time_st, ls="-", color="blue")
        ax2.axvline(val_time_ed, ls="-", color="blue")
        ax2.axvline(val_time_st, ls="-", color="red")
        ax2.axvline(val_time_ed, ls="-", color="red")
        test = np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1)
        ax3.plot(f, np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1), color='blue')
        ax3.plot(f, np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1), color='red')
        ax3.plot(f, np.mean(Zxx[:, idx_time_st:idx_time_ed])-np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1), color='green')
        #ax3.plot(f, np.mean(Zxx[:, :20000], axis=1))
        # plt.contourf(t, f, np.abs(Zxx), 200, norm=LogNorm())# vmax=1e-7)
        ax1.set_ylabel("Frequency [Hz]")
        ax2.set_xlabel("Time [sec]")
        ax3.set_xlabel("Frequency [Hz]")
        #ax3.set_yscale('log')
        #plt.ylim([0, MAXFREQ])
        ax2.set_xlim([np.min(x), np.max(x)])
        ax3.set_xlim([np.min(f), np.max(f)])
        plt.show()

    def interactive_analysis(self):
        print("Enter 'q' then finish:")
        print("'q': quit")
        print("'shoNo'(sn): Shot number")
        print("'Load'(l): Load data")
        print("'SetTime'(st): Set 2 pair of time windows to show spectrum")
        print("'SetShowTime'(sst): Set time window to plot figure")
        print("'stft': Calc STFT")
        print("'pstft': Plot STFT figure")
        print("'fft': Plot STFT figure")
        val_time_show_st = 3.99
        val_time_show_ed = 4.07
        val_time_st = 4.00
        val_time_ed = 4.01
        val_time2_st = 4.02
        val_time2_ed = 4.03
        while True:
            val = input('Enter command: ')
            try:
                if val == 'q':
                    print("FINISH")
                    break
                if val == 'shotNo' or str.upper(val) == 'SN':
                    shotNo = input('Enter shot number: ')

                if val == 'Load' or str.upper(val) == 'L':
                    voltdata, timedata = self.load_data(shotNo=int(shotNo))
                if val == 'SetTime' or str.upper(val) == 'ST':
                    #val_time_st, val_time_ed, val_time2_st, val_time2_ed
                    val_time_st = input('Enter val_time_st: ')
                    val_time_ed = input('Enter val_time_ed: ')
                    val_time2_st = input('Enter val_time2_st: ')
                    val_time2_ed = input('Enter val_time2_ed: ')
                if val == 'SetShowTime' or str.upper(val) == 'SST':
                    val_time_show_st = input('Enter val_time_show_st: ')
                    val_time_show_ed = input('Enter val_time_show_ed: ')

                if val == 'stft':
                    buf_isplot = input('Type "Y" to plot data: ')
                    if str.upper(buf_isplot) == 'Y':
                        isPlot = True
                    else:
                        isPlot = False

                    buf_isIndexing = input('Type "Y" to get index: ')
                    if str.upper(buf_isIndexing) == 'Y':
                        isIndexing = True
                    else:
                        isIndexing = False

                    x, y, f, t, Zxx = self.interactive_stft(int(shotNo), timedata, voltdata,float(val_time_show_st),float(val_time_show_ed),  \
                                          float(val_time_st),float(val_time_ed),float(val_time2_st),float(val_time2_ed), isPlot, isIndexing)

                if val == 'pstft':
                    #self.interactive_stft_plot(int(shotNo), f, t, Zxx)
                    self.interactive_stft_raw_plot(int(shotNo), x, y, f, t, Zxx)
                if val == 'fft':
                    num_plot = input('Input number of plot: ')
                    y_range = input('Input range of y-axis (0 for Autorange): ')
                    buf_isLog = input('Type "Y" to set log scale for y-axis: ')
                    isLog = False
                    if str.upper(buf_isLog) == 'Y':
                        isLog = True
                    self.interactive_fft(int(shotNo), f, t, Zxx, int(num_plot), float(y_range), isLog)

                if val == 'plot':
                    plt.clf()
                    plt.title('#%s' % shotNo, loc='right')
                    plt.plot(timedata[::2000], voltdata[::2000])
                    try:
                        plt.axvline(float(val_time_show_st), ls="-", color="black")
                        plt.axvline(float(val_time_show_ed), ls="-", color="black")
                        plt.axvline(float(val_time_st), ls="-", color="blue")
                        plt.axvline(float(val_time_ed), ls="-", color="blue")
                        plt.axvline(float(val_time2_st), ls="-", color="red")
                        plt.axvline(float(val_time2_ed), ls="-", color="red")
                    except:
                        print("Enter all val_time")
                    plt.xlabel("Time [sec]")
                    plt.show()

                if val == 'help':
                    print("Enter 'q' then finish:")
                    print("'q': quit")
                    print("'shoNo'(sn): Shot number")
                    print("'Load'(l): Load data")
                    print("'SetTime'(st): Set 2 pair of time windows to show spectrum")
                    print("'SetShowTime'(sst): Set time window to plot figure")
                    print("'stft': Calc STFT")
                    print("'pstft': Plot STFT figure")
                    print("'fft': Plot STFT figure")

                if val == 'value':
                    print('shotNo:%d' % int(shotNo))
                    print('Time:%.4f-%.4f, %.4f-%.4f' % (float(val_time_st), float(val_time_ed), float(val_time2_st), float(val_time2_ed)))
                    print('Showing Time:%.4f-%.4f' % (float(val_time_show_st), float(val_time_show_ed)))

                else:
                    print(val)
            except Exception as e:
                print(e)

def getNearestIndex(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    return np.abs(np.asarray(list) - num).argmin()

if __name__ == "__main__":
    start = time.time()
    ctsa = CTS_Analysis()
    #ctsa.stft(shotNo=161898, val_time_st=3.401, val_time_ed=3.404, val_time2_st=3.405, val_time2_ed=3.408, isLocal=True)
    ctsa.interactive_analysis()
    t = time.time() - start
    print("erapsed time= %.3f sec" % t)
