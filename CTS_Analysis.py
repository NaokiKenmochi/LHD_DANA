#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from nifs.voltdata import *
from nifs.timedata import *
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import time

class CTS_Analysis:
    def __init__(self):
        plt.rcParams["font.family"] = "Liberation Sans"  # 全体のフォントを設定
        plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向きへ
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
        plt.rcParams["xtick.major.size"] = 6  # x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = 6  # y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = 3  # x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = 3  # y軸補助目盛り線の長さ
        self.arr_color = ['blue', 'black', 'green','red', 'darkgrey', 'darkorange', 'blue', 'magenta', 'pink']

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
        coef_vmax = float(input('Enter coef_vmax: '))
        val_freq_st = float(input('Enter val_freq_st: '))
        val_freq_ed = float(input('Enter val_freq_ed: '))
        val_time_st = float(input('Enter val_time_st: '))
        val_time_ed = float(input('Enter val_time_ed: '))
        vmax = coef_vmax*np.max(Zxx)
        idx_freq_st = getNearestIndex(f, val_freq_st)
        idx_freq_ed = getNearestIndex(f, val_freq_ed)
        idx_time_st = getNearestIndex(t, val_time_st)
        idx_time_ed = getNearestIndex(t, val_time_ed)

        ncols = 1
        nrows = 5
        grid = GridSpec(nrows, ncols, left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)
        fig = plt.figure(figsize=(8, 6), dpi=150)
        ax1 = fig.add_subplot(grid[0:4, 0])
        plt.title('CTSfosc1 #%d' % shotNo)#, loc='left')
        ax2 = fig.add_subplot(grid[4, 0])
        im0 = ax1.pcolormesh(t[idx_time_st:idx_time_ed], f[idx_freq_st:idx_freq_ed], np.abs(Zxx[idx_freq_st:idx_freq_ed, idx_time_st:idx_time_ed]), vmin=0, vmax=vmax, cmap='inferno')
        ax2.plot(x[::2000], y[::2000])
        ax1.set_ylabel("Frequency [GHz]")
        ax2.set_xlabel("Time [sec]")
        ax2.set_ylabel("CTS [V]")
        ax1.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        ax2.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        # plt.ylim([0, MAXFREQ])
        #ax1.set_xlim([np.min(x), np.max(x)])
        ax2.set_xlim(t[idx_time_st], t[idx_time_ed] + (t[idx_time_ed]-t[idx_time_st])*0.12)
        #plt.colorbar()
        fig.colorbar(im0, ax=ax1, pad=0.01, fraction=0.1)
        plt.show()

        t_end = time.time() - start
        print("erapsed time for plotting STFT = %.3f sec" % t_end)

    def interactive_stft_plot(self, shotNo, f, t, Zxx):
        start = time.time()
        coef_vmax = float(input('Enter coef_vmax: '))
        val_freq_st = float(input('Enter val_freq_st: '))
        val_freq_ed = float(input('Enter val_freq_ed: '))
        val_time_st = float(input('Enter val_time_st: '))
        val_time_ed = float(input('Enter val_time_ed: '))
        vmax = coef_vmax*np.max(Zxx)
        idx_freq_st = getNearestIndex(f, val_freq_st)
        idx_freq_ed = getNearestIndex(f, val_freq_ed)
        idx_time_st = getNearestIndex(t, val_time_st)
        idx_time_ed = getNearestIndex(t, val_time_ed)
        fig = plt.figure(figsize=(8, 6), dpi=150)
        plt.pcolormesh(t[idx_time_st:idx_time_ed], f[idx_freq_st:idx_freq_ed], np.abs(Zxx[idx_freq_st:idx_freq_ed, idx_time_st:idx_time_ed]), vmin=0, vmax=vmax, cmap='inferno')
        plt.title('#%d' % shotNo, loc='right')
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [GHz]")
        plt.colorbar()
        plt.show()

        t_end = time.time() - start
        print("erapsed time for plotting STFT = %.3f sec" % t_end)

    def interactive_stft(self, shotNo, timedata, voltdata, val_time_show_st, val_time_show_ed, val_time_st, val_time_ed, val_time2_st, val_time2_ed, isPlot, isIndexing):
        _,_, noise_spectrum = self.get_digitizer_noise(isLoad=False)
        '''
        nperseg = int(input('Enter nperseg: '))
        noverlap = int(input('Enter noverlap : '))
        nfft = int(input('Enter nfft : '))
        if nperseg == 0:
            nperseg = 2**10
        if noverlap == 0:
            noverlap = nperseg // 8
        if nfft == 0:
            nfft = nperseg
        '''
        nperseg = 2**10
        noverlap = nperseg // 8
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
        #freq_LO = 74e9		#For 77GHz
        freq_LO = 151e9		#For 154GHz

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

            plt.title('#%d' % shotNo, loc='right')
            plt.plot(f, np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1), color='blue')
            plt.plot(f, np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1), color='red')
            Zxx_sub = np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1)-np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1)
            plt.plot(f, Zxx_sub, color='green')
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("PSD [a.u.]")
            plt.yscale('log')
            plt.show()
            #filename = "CTSfosc1_spectrum_" + str(shotNo) + ".txt"
            #np.savetxt(filename, np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1))
            #filename = "CTSfosc1_frequency_GHz_" + str(shotNo) + ".txt"
            #np.savetxt(filename, f)

            coef_vmax = float(input('Enter coef_vmax: '))
            # vmax = 0.05*np.max(Zxx)
            vmax = coef_vmax * np.max(Zxx)
            fig = plt.figure(figsize=(6,8),dpi=150)
            ncols = 4
            nrows = 4
            grid = GridSpec(nrows, ncols, left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)
            fig = plt.figure(figsize=(8,6), dpi=150)
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
            #ax1.pcolormesh(t + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax, cmap='inferno')
            ax1.pcolormesh(t[::100] + time_offset, f, np.abs(Zxx[::1, ::100]), vmin=0, vmax=vmax, cmap='inferno')
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
            Zxx_sub = np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1)-np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1)
            ax3.plot(Zxx_sub, f, color='green')
            ax1.set_ylabel("Frequency [GHz]")
            ax2.set_xlabel("Time [sec]")
            ax3.set_xscale('log')
            #plt.ylim([0, MAXFREQ])
            ax1.set_xlim([np.min(x), np.max(x)])
            ax2.set_xlim([np.min(x), np.max(x)])
            #ax3.set_xlim([0, 1.2*np.max(Zxx_sub)])
            ax3.set_ylim([np.min(f), np.max(f)])
            print('start show')
            #fig.tight_layout()
            plt.show()

            #plt.savefig('plot.png')
            print('end plot')
        '''
        fig = plt.figure(figsize=(6,5), dpi=150)
        title = ('#%d') % (shotNo)
        plt.title(title, loc='right')
        t_probe_on = x[0] + (t[idx_time_ed] + t[idx_time_st])/2
        t_probe_off = x[0] + (t[idx_time2_ed] + t[idx_time2_st])/2
        spectrum_probe_on = np.mean(Zxx[:, idx_time_st:idx_time_ed], axis=1)
        spectrum_probe_off = np.mean(Zxx[:, idx_time2_st:idx_time2_ed], axis=1)
        label_probe_on = "Probe ON, t=%.4f" % t_probe_on
        label_probe_off = "Probe OFF, t=%.4f" % t_probe_off
        label_probe_onmoff = "Probe ON-OFF"
        plt.plot(f, spectrum_probe_on, color='blue', label=label_probe_on)
        plt.plot(f, spectrum_probe_off, color='red', label=label_probe_off)
        plt.plot(f, spectrum_probe_on - spectrum_probe_off, color='green', label=label_probe_onmoff)
        plt.plot(f, noise_spectrum, color='black', label='DIGI NOISE')
        plt.xlabel("Frequency [GHz]")
        plt.legend(frameon=False)
        plt.yscale('log')
        plt.xlim([np.min(f), np.max(f)])
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()
        fig = plt.figure(figsize=(6,5), dpi=150)
        title = ('SIG/NOISE, %.4f-%.4f #%d') % (t_probe_on, t_probe_off, shotNo)
        plt.title(title, loc='right')
        plt.plot(f, (spectrum_probe_on-spectrum_probe_off)/noise_spectrum, color='black')
        plt.xlabel("Frequency [GHz]")
        plt.yscale('log')
        plt.xlim([np.min(f), np.max(f)])
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()
        fig = plt.figure(figsize=(6,5), dpi=150)
        title = ('SIG/(BG-NOISE), %.4f-%.4f #%d') % (t_probe_on, t_probe_off, shotNo)
        plt.title(title, loc='right')
        plt.plot(f, (spectrum_probe_on-spectrum_probe_off)/(spectrum_probe_off-noise_spectrum), color='red')
        plt.xlabel("Frequency [GHz]")
        plt.yscale('log')
        plt.xlim([np.min(f), np.max(f)])
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()
        filename = "CTSfosc1_spectrum_pon_poff_SIGperBGmNOISE_" + str(shotNo)
        np.savez_compressed(filename, f, t, t_probe_on, t_probe_off, spectrum_probe_on, spectrum_probe_off, (spectrum_probe_on-spectrum_probe_off)/(spectrum_probe_off-noise_spectrum))
        '''
        t_end = time.time() - start
        print("erapsed time for STFT = %.3f sec" % t_end)

        t += time_offset
        return x, y, f, t, Zxx

    def load_spectrums(self):
        arr_shotNo = [164449, 164450, 164451, 164452, 164452]
        arr_rho = [0, 0.26, 0.57, 0, 0]
        fig = plt.figure(figsize=(6,5), dpi=150)
        #title = ('SIG/(BG-NOISE), %.4f-%.4f #%d') % (t_probe_on, t_probe_off, shotNo)
        plt.title("CTSfosc1", loc='right')
        for i, idx in enumerate(arr_shotNo):
            filename = "CTSfosc1_spectrum_pon_poff_SIGperBGmNOISE_" + str(idx) + ".npz"
            if i == len(arr_rho)-1:
                filename = "CTSfosc1_spectrum_pon_poff_SIGperBGmNOISE_" + str(idx) + "_v2.npz"

            npz_comp = np.load(filename)
            f = npz_comp['arr_0']
            t = npz_comp['arr_1']
            t_probe_on = npz_comp['arr_2']
            t_probe_off = npz_comp['arr_3']
            spectrum_probe_on = npz_comp['arr_4']
            spectrum_probe_off = npz_comp['arr_5']
            spectrum_probe_SIGperBGmNOISE = npz_comp['arr_6']
            label = '#%d, r/a=%.2f' % (arr_shotNo[i],arr_rho[i])
            plt.plot(f, spectrum_probe_SIGperBGmNOISE, color=self.arr_color[i], label=label)
        plt.legend(frameon=False)
        plt.xlabel("Frequency [GHz]")
        plt.yscale('log')
        plt.xlim([np.min(f), np.max(f)])
        plt.ylim(0.01, 100)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()

    def load_spectrum(self, shotNo):
        filename = "CTSfosc1_spectrum_pon_poff_SIGperBGmNOISE_" + str(shotNo) + ".npz"
        npz_comp = np.load(filename)
        f = npz_comp['arr_0']
        t = npz_comp['arr_1']
        t_probe_on = npz_comp['arr_2']
        t_probe_off = npz_comp['arr_3']
        spectrum_probe_on = npz_comp['arr_4']
        spectrum_probe_off = npz_comp['arr_5']
        spectrum_probe_SIGperBGmNOISE = npz_comp['arr_6']
        fig = plt.figure(figsize=(6,5), dpi=150)
        title = ('SIG/(BG-NOISE), %.4f-%.4f #%d') % (t_probe_on, t_probe_off, shotNo)
        plt.title(title, loc='right')
        plt.plot(f, spectrum_probe_SIGperBGmNOISE, color='red')
        plt.xlabel("Frequency [GHz]")
        plt.yscale('log')
        plt.xlim([np.min(f), np.max(f)])
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()

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

    def get_digitizer_noise(self, isLoad=False):
        shotNo = 164269 #shot number for getting digitizer noise
        if isLoad:
            nperseg = int(input('Enter nperseg: '))
            noverlap = int(input('Enter noverlap : '))
            nfft = int(input('Enter nfft : '))
            if nperseg == 0:
                nperseg = 2 ** 12
            if noverlap == 0:
                noverlap = nperseg // 8
            if nfft == 0:
                nfft = nperseg
            y, x = self.load_data(shotNo=int(shotNo))
            time_offset = x[0]
            freq_LO = 74e9

            MAXFREQ = 1e0
            N = 1e0 * np.abs(1 / (x[1] - x[2]))
            print('start spectgram')
            f, t, Zxx = sig.spectrogram(y, fs=N, nperseg=nperseg, noverlap=noverlap, nfft=nfft)#, window='hamming', nperseg=nperseg, mode='complex')
            spectrum = np.mean(Zxx, axis=1)
            f += freq_LO
            f /= 1e9
            filename = "Data/CTSfosc1_digi_noise_" + str(shotNo)
            np.savez_compressed(filename, f, t, spectrum)

        else:
            filename = "Data/CTSfosc1_digi_noise_" + str(shotNo) + ".npz"
            npz_comp = np.load(filename)
            f = npz_comp['arr_0']
            t = npz_comp['arr_1']
            spectrum = npz_comp['arr_2']

        return f, t, spectrum






    def interactive_analysis(self):
        print("Enter 'q' then finish:")
        print("'q': quit")
        print("'shoNo'(sn): Shot number")
        print("'Load'(l): Load data")
        print("'SetTime'(st): Set 2 pair of time windows to show spectrum")
        print("'SetShowTime'(sst): Set time window to plot figure")
        print("'plot'(p): Plot raw data")
        print("'stft': Calc STFT")
        print("'pstft': Plot STFT figure")
        print("'fft': Plot STFT figure")
        print("'value': Show parameters")
        print("'SaveValue'(sv): Save values")
        print("'LoadValue'(lv): Load values")
        #val_time_show_st = 3.48
        #val_time_show_ed = 3.56
        #val_time_st = 3.505
        #val_time_ed = 3.56
        #val_time2_st = 3.48
        #val_time2_ed = 3.49
        val_time_show_st = 4.00
        val_time_show_ed = 4.08
        val_time_st = 4.058
        val_time_ed = 4.070
        val_time2_st = 4.035
        val_time2_ed = 4.04
        #val_time_show_st = 6.43
        #val_time_show_ed = 6.51
        #val_time_st = 6.43
        #val_time_ed = 6.49
        #val_time2_st = 6.502
        #val_time2_ed = 6.51
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
                    val_time_st = input('Enter val_time_st (Probe ON): ')
                    val_time_ed = input('Enter val_time_ed (Prove ON): ')
                    val_time2_st = input('Enter val_time2_st (Probe OFF): ')
                    val_time2_ed = input('Enter val_time2_ed (Probe OFF): ')
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

                if val == 'plot' or str.upper(val) == 'P':
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
                    plt.xlim(timedata[0], timedata[-1])
                    plt.show()

                if val == 'help':
                    print("Enter 'q' then finish:")
                    print("'q': quit")
                    print("'shoNo'(sn): Shot number")
                    print("'Load'(l): Load data")
                    print("'SetTime'(st): Set 2 pair of time windows to show spectrum")
                    print("'SetShowTime'(sst): Set time window to plot figure")
                    print("'plot'(p): Plot raw data")
                    print("'stft': Calc STFT")
                    print("'pstft': Plot STFT figure")
                    print("'fft': Plot STFT figure")
                    print("'value': Show parameters")
                    print("'SaveValue'(sv): Save values")
                    print("'LoadValue'(lv): Load values")

                if val == 'value':
                    print('shotNo:%d' % int(shotNo))
                    print('Time:%.4f-%.4f, %.4f-%.4f' % (float(val_time_st), float(val_time_ed), float(val_time2_st), float(val_time2_ed)))
                    print('Showing Time:%.4f-%.4f' % (float(val_time_show_st), float(val_time_show_ed)))

                if val == 'SaveValue' or str.upper(val) == 'SV':
                    array_value = np.zeros(7)
                    array_value[0] = shotNo
                    array_value[1] = val_time_st
                    array_value[2] = val_time_ed
                    array_value[3] = val_time2_st
                    array_value[4] = val_time2_ed
                    array_value[5] = val_time_show_st
                    array_value[6] = val_time_show_ed
                    np.savez('value_CTS_Analysis', array_value)

                if val == 'LoadValue' or str.upper(val) == 'LV':
                    array_value = np.load('value_CTS_Analysis.npz')
                    shotNo = array_value['arr_0'][0]
                    val_time_st = array_value['arr_0'][1]
                    val_time_ed = array_value['arr_0'][2]
                    val_time2_st = array_value['arr_0'][3]
                    val_time2_ed = array_value['arr_0'][4]
                    val_time_show_st = array_value['arr_0'][5]
                    val_time_show_ed = array_value['arr_0'][6]

                if val == 'SaveValue2' or str.upper(val) == 'SV2':
                    array_value = np.zeros(7)
                    array_value[0] = shotNo
                    array_value[1] = val_time_st
                    array_value[2] = val_time_ed
                    array_value[3] = val_time2_st
                    array_value[4] = val_time2_ed
                    array_value[5] = val_time_show_st
                    array_value[6] = val_time_show_ed
                    np.savez('value_CTS_Analysis_2', array_value)

                if val == 'LoadValue2' or str.upper(val) == 'LV2':
                    array_value = np.load('value_CTS_Analysis_2.npz')
                    shotNo = array_value['arr_0'][0]
                    val_time_st = array_value['arr_0'][1]
                    val_time_ed = array_value['arr_0'][2]
                    val_time2_st = array_value['arr_0'][3]
                    val_time2_ed = array_value['arr_0'][4]
                    val_time_show_st = array_value['arr_0'][5]
                    val_time_show_ed = array_value['arr_0'][6]

                else:
                    print(val)
            except Exception as e:
                print(e)

    def read_mwscat(self, i_ch):
        shot_num = 181857
        if i_ch < 24:
            digi_name = "mwscat"
            ch_num = i_ch + 9
        elif i_ch >= 24:
            digi_name = "mwscat2"
            ch_num = i_ch + 1 - 24
        sig = RawData.retrieve(digi_name, int(shot_num), 1, ch_num)
        sig_volt = sig.params['RangeHigh'] * 2. / 2 ** sig.params['Resolution(bit)']
        sig_voltdata = sig.get_val() * sig_volt
        print(i_ch)
        plt.plot(sig_voltdata)
        plt.show()
        #return sig_voltdata

    def multiplocess_mwscat(self):
        with Pool(32) as p:
            #arr = p.map(self.read_mwscat, range(32))
            p.map(self.read_mwscat, range(32))


        #return arr

    def plot_mwscat(self):
        shot_num = 179949
        for i in range(32):
            if i < 24:
                digi_name = "mwscat"
                ch_num = i + 9
            elif i >= 24:
                digi_name = "mwscat2"
                ch_num = i + 1 - 24
            sig = RawData.retrieve(digi_name, int(shot_num), 1, ch_num)
            sig_volt = sig.params['RangeHigh'] * 2. / 2 ** sig.params['Resolution(bit)']
            sig_voltdata = sig.get_val() * sig_volt
            print(i)

def getNearestIndex(list, num):
    """
    概要: リストからある値に    最も近い値を返却する関数
    @param list:     データ配列
    @param num: 対    象値
    @return 対象値に最    も近い値
    """

    # リスト要素と対象値の差    分を計算し最小値のインデックスを取得
    return np.abs(np.asarray(list) - num).argmin()

if __name__ == "__main__":
    start = time.time()
    ctsa = CTS_Analysis()
    #ctsa.stft(shotNo=161898, val_time_st=3.401, val_time_ed=3.404, val_time2_st=3.405, val_time2_ed=3.408, isLocal=True)
    ctsa.interactive_analysis()
    #ctsa.multiplocess_mwscat()
    #ctsa.plot_mwscat()
    #ctsa.load_spectrum(164452)
    #ctsa.load_spectrums()
    #ctsa.get_digitizer_noise(isLoad=True)
    t = time.time() - start
    print("erapsed time= %.3f sec" % t)
