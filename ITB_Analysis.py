import sys
import os
import csv
import pywt
from nifs.anadata import *
from nifs.voltdata import *
from nifs.timedata import *
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from ftplib import FTP
from scipy import fftpack
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.interpolate import interp1d, griddata
from scipy.signal import find_peaks, peak_prominences, butter, filtfilt, savgol_filter, coherence, hilbert, firwin
from scipy.fftpack import fft, ifft, fftfreq
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from scipy import signal, integrate
import scipy.signal as sig
import numpy as np
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import statistics
#import pandas as pd
from matplotlib.colors import LogNorm
from decimal import Decimal
from tqdm import tqdm  # 進捗バーを表示するためのライブラリをインポート
#from matplotlib.animation import ArtistAnimation
from PIL import Image, ImageDraw


class ITB_Analysis:
    def __init__(self, ShotNo, ShotNo_2=None):
        plt.rcParams["font.family"] = "Liberation Sans"  # 全体のフォントを設定
        plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向きへ
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
        plt.rcParams["xtick.major.size"] = 6  # x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = 6  # y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = 3  # x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = 3  # y軸補助目盛り線の長さ
        self.arr_color = ['black', 'darkgrey', 'red', 'darkorange', 'green', 'blue', 'magenta', 'pink']

        self.ShotNo = ShotNo
        #self.ShotNo = np.int(sys.argv[1])
        self.ShotNo_2 = ShotNo_2

    def ana_plot_allThomson_sharex(self):

        data = AnaData.retrieve('thomson', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        #dnum = data.getDimNo()
        #for i in range(dnum):
        #    print(i, data.getDimName(i), data.getDimUnit(i))
        #vnum = data.getValNo()
        #for i in range(vnum):
        #    print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        reff = data.getDimData('R')
        r = reff/1000

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        ne = data.getValData('n_e')
        dne = data.getValData('dn_e')

        # 各時刻毎の R 対 Te のデータをプロット

        #plt.figure(figsize=(5.3, 3), dpi=100)
        #plt.xlabel('reff/a99')
        #plt.ylabel('iota, Te [keV], ne [e19m-3]')
        labels = []


        #f, axarr = plt.subplots(2, len(t), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(1.5*len(t),3), dpi=100)
        f, axarr = plt.subplots(2, 100, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(1.5*100,3), dpi=200)

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            if i % 100 != 0:
                ax.set_yticklabels([])
            if i < 100:
                ax.set_xticklabels([])
        # データ取得
        title = ('#%d\n') % (self.ShotNo)
        axarr[0,0].set_title(title, loc='left')
        #for idx in range(len(t)):
        for idx in range(100):
            T = Te[idx+90, :]/1000  # R 次元でのスライスを取得
            dT = dTe[idx+90, :]/1000  # R 次元でのスライスを取得
            n = ne[idx+90, :]/1000  # R 次元でのスライスを取得
            dn = dne[idx+90, :]/1000  # R 次元でのスライスを取得
            axarr[1,idx].errorbar(r, T, yerr=dT, capsize=2, fmt='o', markersize=2, ecolor='red', markeredgecolor='red', color='w')
            axarr[0,idx].errorbar(r, n, yerr=dn, capsize=2, fmt='o', markersize=2, ecolor='blue', markeredgecolor='blue', color='w')
            #axarr[0,idx].set_xlim(-1.2, 1.2)
            #axarr[1,idx].set_xlim(-1.2, 1.2)
            axarr[0,idx].set_ylim(0, 5.5)
            axarr[1,idx].set_ylim(0, 5.5)
            title_time = ('t=%.3fsec') % (t[idx+90]/1000)
            axarr[0,idx].set_title(title_time)
            axarr[1, idx].set_xlabel('R [m]')


        axarr[0, 0].set_ylabel(r'$n_e [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e [keV]$')


        plt.savefig("Thomson_Te_ne_No%d.png" % self.ShotNo)
        plt.show()
        #plt.close()

    def plot_TS_Te_vs_ne(self):

        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)
        #data = AnaData.retrieve('thomson', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
        #dnum = data.getDimNo()
        #for i in range(dnum):
        #    print(i, data.getDimName(i), data.getDimUnit(i))
        #vnum = data.getValNo()
        #for i in range(vnum):
        #    print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        #reff = data.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        #ne = data.getValData('n_e')
        #dne = data.getValData('dn_e')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        #Te_fit = data.getValData('Te_fit')
        #ne_fit = data.getValData('ne_fit')
        plt.plot(t, np.mean(Te[:,53:56], axis=1))
        plt.plot(t, np.mean(ne[:,53:56], axis=1))
        plt.ylim(0,5)
        plt.xlim(3,10)
        plt.show()
        t_ne_Te = np.vstack((t, np.mean(ne[:,53:56], axis=1), np.mean(Te[:,53:56], axis=1)))
        filename = 'time_ne0_Te0_w3av_SN%d.txt' % (self.ShotNo)
        np.savetxt(filename, t_ne_Te.T)
        f, axarr = plt.subplots(2, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 10), dpi=150, sharex=True)

        axarr[0].set_title(self.ShotNo, fontsize=18, loc='right')
        axarr[0].set_ylabel(r'$T_e$ [keV]', fontsize=18)
        axarr[1].set_xlabel('Time [sec]', fontsize=18)
        axarr[1].set_ylabel(r'$dT_e$ [keV]', fontsize=18)
        axarr[1].set_xlabel(r'$n_e\ [10^{19}m^{-3}]$', fontsize=14)
        axarr[0].set_ylim(0, 15)
        axarr[1].set_ylim(0, 3)
        axarr[1].set_xlim(0, 2)
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[0].plot(ne[:,53], Te[:,::10], "o")
        axarr[1].plot(ne[:,53], dTe[:,::10], "o")
        plt.savefig("Te_vs_ne_TS_No%d.png" % self.ShotNo)
        plt.show()

    def ana_plot_allTS_sharex(self):

        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        #dnum = data.getDimNo()
        #for i in range(dnum):
        #    print(i, data.getDimName(i), data.getDimUnit(i))
        #vnum = data.getValNo()
        #for i in range(vnum):
        #    print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')

        # 各時刻毎の R 対 Te のデータをプロット

        #plt.figure(figsize=(5.3, 3), dpi=100)
        #plt.xlabel('reff/a99')
        #plt.ylabel('iota, Te [keV], ne [e19m-3]')
        labels = []


        f, axarr = plt.subplots(3, len(t), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(1.5*len(t),5), dpi=100)

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            if i % len(t) != 0:
                ax.set_yticklabels([])
            if i < 2*len(t):
                ax.set_xticklabels([])
        # データ取得
        title = ('#%d\n') % (self.ShotNo)
        axarr[0,0].set_title(title, loc='left')
        for idx in range(len(t)):
            r = reff[idx, :]
            T = Te[idx, :]  # R 次元でのスライスを取得
            dT = dTe[idx, :]  # R 次元でのスライスを取得
            n = ne[idx, :]  # R 次元でのスライスを取得
            dn = dne[idx, :]  # R 次元でのスライスを取得
            T_fit = Te_fit[idx, :]  # R 次元でのスライスを取得
            n_fit = ne_fit[idx, :]  # R 次元でのスライスを取得
            #axarr[1,idx].plot(r, T_fit,  '-', color='red')
            #axarr[0,idx].plot(r, n_fit,  '-', color='blue')
            axarr[1,idx].errorbar(r, T, yerr=dT, capsize=2, fmt='o', markersize=2, ecolor='red', markeredgecolor='red', color='w')
            axarr[0,idx].errorbar(r, n, yerr=dn, capsize=2, fmt='o', markersize=2, ecolor='blue', markeredgecolor='blue', color='w')
            axarr[0,idx].set_xlim(-1.2, 1.2)
            axarr[1,idx].set_xlim(-1.2, 1.2)
            axarr[2,idx].set_xlim(-1.2, 1.2)
            axarr[0,idx].set_ylim(0, 2.5)
            axarr[1,idx].set_ylim(0, 15.5)
            title_time = ('t=%.3fsec') % (t[idx])
            axarr[0,idx].set_title(title_time)
            axarr[2, idx].set_xlabel(r'$r_{eff}/a_{99}$')

        try:
            data_lhdgauss = AnaData.retrieve('LHDGAUSS_DEPROF', self.ShotNo, 1)

            reff_lhdgauss = data_lhdgauss.getDimData('reff')

            t_lhdgauss = data_lhdgauss.getDimData('time')

            sum_power_density = data_lhdgauss.getValData('Sum_Power_Density')
            sum_total_power = data_lhdgauss.getValData('Sum_Total_Power')
            idx_arr_time_lhdgauss = find_closest(t_lhdgauss, t)
            for i, idx in enumerate(idx_arr_time_lhdgauss):
                sum_pd = sum_power_density[idx, :]
                sum_tp = sum_total_power[idx, :]
                axarr[2,i].plot(-reff_lhdgauss, sum_pd/2, '-', label='Power_Density', color='gray')
                axarr[2,i].plot(-reff_lhdgauss, sum_tp/2, '--', label='Total_Power', color='black')
                axarr[2,i].set_ylim(0, 1.2)
        except Exception as e:
            print(e)

        try:
            data_mse = AnaData.retrieve('lhdmse1_iota_ms', self.ShotNo, 1)
            t_mse = data_mse.getDimData('t')
            reff_mse = data_mse.getValData('reff/a99')
            iota = data_mse.getValData('iota')
            iota_vac = data_mse.getValData('iota_vac')
            iota_data_ip = data_mse.getValData('iota_data_ip')
            iota_data_error_ip = data_mse.getValData('iota_data_error_ip')
            idx_arr_time = find_closest(t, t_mse)
            for i, idx in enumerate(idx_arr_time):
                r_1d = reff_mse[i, :]
                iota_1d = iota[i, :]  # R 次元でのスライスを取得
                iota_vac_1d = iota_vac[i, :]  # R 次元でのスライスを取得
                iota_data_ip_1d = iota_data_ip[i, :]  # R 次元でのスライスを取得
                iota_data_error_ip_1d = iota_data_error_ip[i, :]  # R 次元でのスライスを取得
                axarr[2,idx].plot(r_1d, iota_1d, '-', label='iota', color='green')
                axarr[2,idx].errorbar(r_1d, iota_data_ip_1d, yerr=iota_data_error_ip_1d, capsize=2, fmt='o', markersize=2, ecolor='green', markeredgecolor='green', color='w')
                axarr[2,idx].plot(r_1d, iota_vac_1d, '--', label='iota_vac', color='gray')
        except Exception as e:
            print(e)

        axarr[0, 0].set_ylabel(r'$n_e [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e [keV]$')
        axarr[2, 0].set_ylabel(r'$\iota/2\pi$')


        plt.savefig("Te_ne_iota_No%d.png" % self.ShotNo)
        #plt.show()
        plt.close()


    def ana_plot_allMSE_sharex(self):

        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)
        data_mse = AnaData.retrieve('lhdmse1_iota_ms', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        t_mse = data_mse.getDimData('t')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')


        labels = []


        f, axarr = plt.subplots(3, len(t_mse), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(1.5*len(t_mse),5), dpi=150)

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            if i % len(t_mse) != 0:
                ax.set_yticklabels([])
            if i < 2*len(t_mse):
                ax.set_xticklabels([])
        # データ取得
        title = ('#%d\n') % (self.ShotNo)
        axarr[0,0].set_title(title, loc='left', fontsize=16)
        idx_arr_time = find_closest(t, t_mse)
        for i, idx in enumerate(idx_arr_time):
            r = reff[idx, :]
            T = Te[idx, :]  # R 次元でのスライスを取得
            dT = dTe[idx, :]  # R 次元でのスライスを取得
            n = ne[idx, :]  # R 次元でのスライスを取得
            dn = dne[idx, :]  # R 次元でのスライスを取得
            #T_fit = Te_fit[idx, :]  # R 次元でのスライスを取得
            #n_fit = ne_fit[idx, :]  # R 次元でのスライスを取得
            #axarr[1,i].plot(r, T_fit,  '-', color='red')
            axarr[1,i].errorbar(r, T, yerr=dT, capsize=2, fmt='o', markersize=2, ecolor='red', markeredgecolor='red', color='w')
            axarr[0,i].errorbar(r, n, yerr=dn, capsize=2, fmt='o', markersize=2, ecolor='blue', markeredgecolor='blue', color='w')
            #axarr[0,i].plot(r, n_fit,  '-', color='blue')
            axarr[0,i].set_xlim(-1.2, 1.2)
            axarr[1,i].set_xlim(-1.2, 1.2)
            axarr[2,i].set_xlim(-1.2, 1.2)
            axarr[0,i].set_ylim(0, 2.2)
            axarr[1,i].set_ylim(0, 17)
            axarr[0, i].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[1, i].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[2, i].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            title_time = ('t=%.3fsec') % (t[idx])
            axarr[0,i].set_title(title_time)
            axarr[2, i].set_xlabel(r'$r_{eff}/a_{99}$')

        '''
        try:
            data_lhdgauss = AnaData.retrieve('LHDGAUSS_DEPROF', self.ShotNo, 1)

            reff_lhdgauss = data_lhdgauss.getDimData('reff')

            t_lhdgauss = data_lhdgauss.getDimData('time')

            sum_power_density = data_lhdgauss.getValData('Sum_Power_Density')
            sum_total_power = data_lhdgauss.getValData('Sum_Total_Power')
            idx_arr_time_lhdgauss = find_closest(t_lhdgauss, t_mse)
            for i, idx in enumerate(idx_arr_time_lhdgauss):
                sum_pd = sum_power_density[idx, :]
                sum_tp = sum_total_power[idx, :]
                axarr[2,i].plot(-reff_lhdgauss, sum_pd/2, '-', label='Power_Density', color='gray')
                axarr[2,i].plot(-reff_lhdgauss, sum_tp/2, '--', label='Total_Power', color='black')
                axarr[2,i].set_ylim(0, 1.2)
        except Exception as e:
            print(e)
        '''

        reff_mse = data_mse.getValData('reff/a99')
        iota = data_mse.getValData('iota')
        iota_vac = data_mse.getValData('iota_vac')
        iota_data_ip = data_mse.getValData('iota_data_ip')
        iota_data_error_ip = data_mse.getValData('iota_data_error_ip')
        for i in range(len(t_mse)):
            r_1d = reff_mse[i, :]
            iota_1d = iota[i, :]  # R 次元でのスライスを取得
            iota_vac_1d = iota_vac[i, :]  # R 次元でのスライスを取得
            iota_data_ip_1d = iota_data_ip[i, :]  # R 次元でのスライスを取得
            iota_data_error_ip_1d = iota_data_error_ip[i, :]  # R 次元でのスライスを取得
            axarr[2,i].plot(r_1d, iota_1d, '-', label='iota', color='green')
            axarr[2,i].errorbar(r_1d, iota_data_ip_1d, yerr=iota_data_error_ip_1d, capsize=2, fmt='o', markersize=2, ecolor='green', markeredgecolor='green', color='w')
            axarr[2,i].plot(r_1d, iota_vac_1d, '--', label='iota_vac', color='gray')
            axarr[2,i].set_ylim(0, 1.2)

        axarr[0, 0].set_ylabel(r'$n_e\ [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e\ [keV]$')
        axarr[2, 0].set_ylabel(r'$\iota/2\pi$')
        f.align_labels()


        plt.savefig("Te_ne_iota_allMSE_No%d.png" % self.ShotNo)
        #plt.show()
        #plt.close()

    def ana_plot_eg_sharex(self):

        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)
        data_mse = AnaData.retrieve('lhdmse1_iota_ms', self.ShotNo, 1)
        data_lhdgauss = AnaData.retrieve('LHDGAUSS_DEPROF', self.ShotNo, 1)

        reff_lhdgauss = data_lhdgauss.getDimData('reff')

        t_lhdgauss = data_lhdgauss.getDimData('time')

        sum_power_density = data_lhdgauss.getValData('Sum_Power_Density')
        sum_total_power = data_lhdgauss.getValData('Sum_Total_Power')

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
        dnum = data_mse.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data_mse.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        t_mse = data_mse.getDimData('t')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        reff_mse = data_mse.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        iota = data_mse.getValData('iota')
        iota_vac = data_mse.getValData('iota_vac')
        iota_data_ip = data_mse.getValData('iota_data_ip')
        iota_data_error_ip = data_mse.getValData('iota_data_error_ip')

        # 各時刻毎の R 対 Te のデータをプロット

        #plt.figure(figsize=(5.3, 3), dpi=100)
        #plt.xlabel('reff/a99')
        #plt.ylabel('iota, Te [keV], ne [e19m-3]')
        labels = []

        idx_arr_time = find_closest(t, t_mse)
        idx_arr_time_lhdgauss = find_closest(t_lhdgauss, t_mse)
        val_arr_time = t[idx_arr_time]

        plt.clf()
        f, axarr = plt.subplots(3, len(t_mse), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(2*len(t_mse),6), dpi=100)

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            if i % len(t_mse) != 0:
                ax.set_yticklabels([])
            if i < 2*len(t_mse):
                ax.set_xticklabels([])
        # データ取得
        title = ('#%d\n') % (self.ShotNo)
        axarr[0,0].set_title(title, loc='left')
        for i, idx in enumerate(idx_arr_time):
            r = reff[idx, :]
            T = Te[idx, :]  # R 次元でのスライスを取得
            dT = dTe[idx, :]  # R 次元でのスライスを取得
            n = ne[idx, :]  # R 次元でのスライスを取得
            dn = dne[idx, :]  # R 次元でのスライスを取得
            T_fit = Te_fit[idx, :]  # R 次元でのスライスを取得
            n_fit = ne_fit[idx, :]  # R 次元でのスライスを取得
            axarr[1,i].plot(r, T_fit,  '-', color='black')
            axarr[1,i].errorbar(r, T, yerr=dT, capsize=2, fmt='o', markersize=2, ecolor='black', markeredgecolor='black', color='w')
            axarr[0,i].errorbar(r, n, yerr=dn, capsize=2, fmt='o', markersize=2, ecolor='black', markeredgecolor='black', color='w')
            axarr[0,i].plot(r, n_fit,  '-', color='black')
            #axarr[1,i].set_xlim(-1.2, 1.2)
            axarr[0,i].set_xlim(-1.2, 1.2)
            axarr[0,i].set_ylim(0, 1.5)
            axarr[1,i].set_ylim(0, 10.5)
            title_time = ('t=%.3fsec') % (t_mse[i])
            axarr[0,i].set_title(title_time)

        for i, idx in enumerate(idx_arr_time_lhdgauss):
            sum_pd = sum_power_density[idx, :]
            sum_tp = sum_total_power[idx, :]
            axarr[2,i].plot(-reff_lhdgauss, sum_pd/2, '-', label='Power_Density', color='gray')
            axarr[2,i].plot(-reff_lhdgauss, sum_tp/2, '--', label='Total_Power', color='black')

        for i in range(len(t_mse)):
            r_1d = reff_mse[i, :]
            iota_1d = iota[i, :]  # R 次元でのスライスを取得
            iota_vac_1d = iota_vac[i, :]  # R 次元でのスライスを取得
            iota_data_ip_1d = iota_data_ip[i, :]  # R 次元でのスライスを取得
            iota_data_error_ip_1d = iota_data_error_ip[i, :]  # R 次元でのスライスを取得
            axarr[2,i].plot(r_1d, iota_1d, '-', label='iota', color='black')
            axarr[2,i].errorbar(r_1d, iota_data_ip_1d, yerr=iota_data_error_ip_1d, capsize=2, fmt='o', markersize=2, ecolor='black', markeredgecolor='black', color='w')
            axarr[2,i].plot(r_1d, iota_vac_1d, '--', label='iota_vac', color='gray')
            #axarr[2,i].set_xlim(-1.2, 1.2)
            axarr[2,i].set_ylim(0, 1.2)
            axarr[2,i].set_xlabel(r'$r_{eff}/a_{99}$')

        axarr[0, 0].set_ylabel(r'$n_e [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e [keV]$')
        axarr[2, 0].set_ylabel(r'$\iota/2\pi$')


        plt.show()
        plt.close()

    def plot_HSTS(self):
        #hs_thomson_buf = eg.load('high_time_res_thomson/high_time_res_thomson_s' + SN + '.dat')
        hs_thomson = AnaData.retrieve('high_time_res_ts_model_fit', self.ShotNo, 1)
        #hs_thomson_buf = eg.load('high_time_res_thomson/high_time_res_thomson_s' + SN + '_v2.dat')
        #tsmap_calib = eg.load('tsmap_calib/tsmap_calib@' + SN + '.txt')
        tsmap_calib = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)
        arr_reffa99_TS = tsmap_calib.getValData('reff/a99')
        #Te = data.getValData('Te')
        #dTe = data.getValData('dTe')
        #ne = data.getValData('ne_calFIR')
        #dne = data.getValData('dne_calFIR')
        #Te_fit = data.getValData('Te_fit')
        #ne_fit = data.getValData('ne_fit')
        arr_R_TS = tsmap_calib.getDimData('R')
        i_time=43
        print(tsmap_calib.getDimData('Time')[i_time])
        arr_reff_HSTS = arr_reffa99_TS[i_time, find_closest(arr_R_TS, hs_thomson.getDimData('R'))]
        #tsmap_smooth = tsmap_smooth_buf.sortby('reff/a99')
        #kernel = np.full((5,5), 1/25) #default
        kernel = np.full((3,3), 1/9)
        #kernel = np.full((1,1), 1/1)
        #kernel = np.full((2,2), 1/4)
        #kernel = np.full((3,2), 1/6)
        #data_HSTS_filtered = apply_kernel(normalize_all_columns_np(hs_thomson.getValData('Te')), kernel)
        col = 40
        data_HSTS_filtered = apply_kernel(normalize_all_columns_np(delete_column(hs_thomson.getValData('Te'), col)), kernel)
        #data_HSTS_filtered = apply_kernel(normalize_all_columns_np(delete_column(hs_thomson.getValData('ne'), col)), kernel)
        #data_HSTS_filtered = apply_kernel(delete_column(hs_thomson.getValData('Te'), col), kernel)
        #data_HSTS_filtered = normalize_all_columns_np(apply_kernel(delete_column(hs_thomson.getValData('Te'), col), kernel))
        index = col
        arr_reff_HSTS_deleted = np.delete(arr_reff_HSTS, index)

        #arr_R_TS = np.array(tsmap_calib['R'])
        #arr_reffa99_TS = np.array(tsmap_calib['reff/a99'].isel(Time=i_time))
        #plt.imshow(hs_thomson['Te'], vmax=10)
        #plt.pcolormesh(hs_thomson.getDimData('Time'), hs_thomson.getDimData('R'), hs_thomson.getValData('Te').T, cmap='jet', vmin=0, vmax=6)
        #plt.pcolormesh(hs_thomson.getDimData('Time'), hs_thomson.getDimData('R'), np.array(normalize_all_columns_np(hs_thomson.getValData('Te'))).T, cmap='jet', vmin=0, vmax=1)
        #plt.pcolormesh(hs_thomson.getDimData('Time'), hs_thomson.getDimData('R'), np.array(normalize_all_columns_np(data_HSTS_filtered)).T, cmap='jet', vmin=0, vmax=1)
        #plt.pcolormesh(hs_thomson.getDimData('Time'), hs_thomson.getDimData('R'), np.array(data_HSTS_filtered).T, cmap='jet')
        #plt.pcolormesh(hs_thomson.getDimData('Time'), arr_reff_HSTS, np.array(data_HSTS_filtered).T, cmap='jet')
        plt.pcolormesh(hs_thomson.getDimData('Time'), arr_reff_HSTS_deleted, np.array(data_HSTS_filtered).T, cmap='jet', vmin=0, vmax=1)
        #plt.pcolormesh(hs_thomson.getDimData('Time'), arr_reff_HSTS, np.array(normalize_all_columns_np(hs_thomson.getValData('Te'))).T, cmap='jet')
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        #plt.title('SN' + str(self.ShotNo))
        plt.title('SN' + str(self.ShotNo) + ', kernel:' + str(kernel.shape) )
        plt.xlabel('Time [sec]', fontsize=15)
        #plt.ylabel('R [m]', fontsize=15)
        plt.ylabel('$r_{eff}/a_{99}$', fontsize=15)
        plt.ylim(-1.2,1.2)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.get_yaxis().labelpad = 19
        #cbar.set_label('$T_e$ [keV]', rotation=270, fontsize=15)
        #cbar.set_label('$T_e$ [keV]', rotation=270, fontsize=15)
        cbar.set_label('$T_e$ [0,1]', rotation=270, fontsize=15)
        #filename = 'HSTS_reff_filtered_normalized_kernel33_%d' % (self.ShotNo)
        filename = 'HSTS_Te_01_%d' % (self.ShotNo)
        plt.savefig(filename)
        plt.show()
        plt.close()

        #data_HSTS_filtered = apply_kernel(delete_column(hs_thomson.getValData('ne'), col), kernel)
        data_HSTS_filtered = normalize_all_columns_np(apply_kernel(delete_column(hs_thomson.getValData('ne'), col), kernel))
        plt.pcolormesh(hs_thomson.getDimData('Time'), arr_reff_HSTS_deleted, np.array(data_HSTS_filtered).T, cmap='jet', vmin=0, vmax=1)
        #plt.pcolormesh(hs_thomson.getDimData('Time'), arr_reff_HSTS, np.array(normalize_all_columns_np(hs_thomson.getValData('Te'))).T, cmap='jet')
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        #plt.title('SN' + str(self.ShotNo))
        plt.title('SN' + str(self.ShotNo) + ', kernel:' + str(kernel.shape) )
        plt.xlabel('Time [sec]', fontsize=15)
        #plt.ylabel('R [m]', fontsize=15)
        plt.ylabel('$r_{eff}/a_{99}$', fontsize=15)
        plt.ylim(-1.2,1.2)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.get_yaxis().labelpad = 19
        #cbar.set_label('$T_e$ [keV]', rotation=270, fontsize=15)
        #cbar.set_label('$n_e$ [m^-3]', rotation=270, fontsize=15)
        cbar.set_label('$n_e$ [0,1]', rotation=270, fontsize=15)
        #filename = 'HSTS_reff_filtered_normalized_kernel33_%d' % (self.ShotNo)
        filename = 'HSTS_ne_01_%d' % (self.ShotNo)
        plt.savefig(filename)
        plt.show()
        plt.close()

        '''
        kernel = np.full((5,5), 1/25) #default
        #kernel = np.full((7,7), 1/49)
        #kernel = np.full((3,5), 1/15)
        #kernel = np.full((1,1), 1/1)
        #hs_thomson = hs_thomson_buf#.sortby('R')
        arr_all_buf = np.delete(np.array(hs_thomson['Te']), [2,3,40,58,59,65,66,67,68,69,70], 1)
        arr_all_dTe_buf = np.delete(np.array(hs_thomson['dTe']), [2,3,40,58,59,65,66,67,68,69,70], 1)
        #arr_all_dTe_buf = np.delete(np.array(hs_thomson['dne']), [2,3,40,58,59,65,66,67,68,69,70], 1)/1e19
        arr_R = np.delete(np.array(hs_thomson['R']), [2,3,40,58,59,65,66,67,68,69,70], 0)
        #arr_all_buf = np.delete(np.array(hs_thomson['Te']), [0], 1)
        #arr_all_buf = np.delete(np.array(hs_thomson['ne']), [0], 1)/1e19
        #arr_R = np.delete(np.array(hs_thomson['R']), [0], 0)
        arr_all = cv2.filter2D(arr_all_buf, -1, kernel)
        arr_all_dTe = arr_all_dTe_buf
        #arr_all_dTe = cv2.filter2D(arr_all_dTe_buf, -1, kernel)
        arr_dT = arr_all[:,1:] - arr_all[:,:-1]#np.array(arr2) - np.array(arr1)
        arr_dTdt = arr_all[1:,:] - arr_all[:-1,:]#np.array(arr2) - np.array(arr1)
        arr_dR = arr_R[1:] - arr_R[:-1]

        arr_reff_HSTS = arr_reffa99_TS[find_closest(arr_R_TS, arr_R)]
        arr_reff_HSTS -= 0.05

        fig = plt.figure(figsize=(8,3), dpi=300)
        plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加

        # Figureと3DAxeS
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        '''

        '''
        x = 4.398907
        y = np.linspace(-1, 1, 11)
        z = np.linspace(0, 7, 11)
        Y, Z = np.meshgrid(y, z)
        X = np.array([x] * Y.shape[0])
        ax.plot_surface(X, Y, Z, alpha=0.3, color='red')
        x = 4.399857
        X = np.array([x] * Y.shape[0])
        ax.plot_surface(X, Y, Z, alpha=0.3, color='blue')
        '''

        '''
        # 軸ラベルを設定
        ax.set_xlabel("Time [s]", size=16)
        ax.set_ylabel("$r_{eff}/a_{99}$", size=16)
        ax.set_zlabel("$T_e$ [keV]", size=16)

        X, Y = np.meshgrid(hs_thomson['Time'][:-2], arr_reff_HSTS)

        # Normalize the colors based on Z value
        norm = plt.Normalize(arr_all[:-2,:].T.min(), arr_all[:-2,:].T.max())
        colors = cm.jet(norm(arr_all[:-2,:].T))
        #norm = plt.Normalize(arr_all[:-2,:].min(), arr_all[:-2,:].max())
        #colors = cm.jet(norm(arr_all[:-2,:]))
        ax = plt.axes(projection='3d')
        #surf = ax.plot_surface(X, Y, arr_all[:-2,:].T, shade=False, facecolors=colors, rstride=100, cstride=5)
        #surf = ax.plot_surface(X, Y, arr_all[:-2,:].T, shade=False, facecolors=colors, rstride=5, cstride=5)
        #surf = ax.plot_surface(Y, X, arr_all[:-2,:].T, shade=False, facecolors=colors, rstride=100, cstride=1)
        #surf.set_facecolor((0, 0, 0, 0))
        # 曲面を描画
        #ax.plot_wireframe(X, Y, arr_all[:-2,:].T, rstride=0, cstride=10)
        #ax.plot_surface(X, Y, arr_all[:-2,:].T, cmap="jet")
        ax.plot_surface(X, Y, 0.77*arr_all[:-2,:].T, cmap="jet")    #温度表示
        ax.set_zlim(0, 5)

        #plt.savefig("HSTS.svg")
        plt.show()

        #plt.pcolormesh(hs_thomson['Time'], hs_thomson['R'], arr_all.T, cmap='jet', vmin=0, vmax=6)
        #plt.pcolormesh(hs_thomson['Time'], arr_R, arr_all.T, cmap='jet', vmin=0, vmax=6)
        plt.pcolormesh(hs_thomson['Time'], arr_reff_HSTS, arr_all.T, cmap='jet', vmin=0, vmax=10)
        #plt.vlines(4.398907, -1, 1, colors='red')
        #plt.vlines(4.399857, -1, 1, colors='blue')
        plt.title('SN' + self.ShotNo, loc='right')
        plt.xlabel('Time [sec]')
        plt.ylim(-1, 1)
        #plt.ylabel('R [m]')
        plt.ylabel('$r_{eff}/a_{99}$', fontsize=15)
        #plt.ylim(2.9, 4.50)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('$T_e$ [keV]', rotation=270)
        plt.show()
        '''

        '''
        fig = plt.figure(figsize=(8,3), dpi=150)
        #plt.pcolormesh(hs_thomson['Time'], hs_thomson['R'], arr_all.T, cmap='jet', vmin=0, vmax=6)
        #plt.pcolormesh(hs_thomson['Time'], arr_R, arr_all.T, cmap='jet', vmin=0, vmax=6)
        plt.pcolormesh(hs_thomson['Time'][1:], arr_reff_HSTS, arr_dTdt.T, cmap='bwr', vmin=-0.2, vmax=0.2)
        plt.title('SN' + self.ShotNo, loc='right')
        plt.xlabel('Time [sec]')
        #plt.ylabel('R [m]')
        plt.ylabel('reff/a99 [m]')
        #plt.ylim(2.9, 4.50)
        cbar = plt.colorbar()
        #cbar.set_label('dTe [keV]', rotation=270)
        plt.show()
        '''


        '''
        fig = plt.figure(figsize=(8,3), dpi=150)
        #arr1 = (hs_thomson['Te'][:,1:])
        #arr2 = hs_thomson['Te'][:,:-1]
        arr_dR_s = np.squeeze(arr_dR)
        arr_dTedR = arr_dT/arr_dR_s
        #plt.pcolormesh(hs_thomson['Time'], arr_R[1:], (arr_dTedR).T, cmap='bwr', vmin=-25, vmax=25)
        plt.pcolormesh(hs_thomson['Time'], arr_reff_HSTS[1:], (arr_dTedR).T, cmap='bwr', vmin=-35, vmax=35)
        #plt.pcolormesh(arr.T, cmap='bwr', vmin=-1.0, vmax=1.0)
        plt.title('SN' + self.ShotNo, loc='right')
        plt.xlabel('Time [sec]')
        #plt.ylabel('R [m]')
        plt.ylabel('$r_{eff}/a_{99}$', fontsize=15)
        plt.ylim(-1, 1)
        #plt.ylim(2.9, 4.50)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('$dT_e/dR$ [keV/m]', rotation=270)
        plt.show()

        #plt.plot(hs_thomson['Time'], arr_all[:,28])
        #plt.show()
        #plt.plot(hs_thomson['Time'], np.abs(arr[:,38]), 'o')
        #plt.plot(hs_thomson['Time'], np.abs(arr[:,35:39]), 'o')
        arr_dR_s = np.squeeze(arr_dR)
        arr_dTedR = np.abs(arr_dT/arr_dR_s)
        fig = plt.figure(figsize=(8,2), dpi=150)
        i_position = 29#28
        label='reff/a99= ' + str(arr_reff_HSTS[i_position])[:]
        plt.plot(hs_thomson['Time'], np.abs(arr_dTedR[:,i_position]), 'bo', label=label)
        i_position = 24#23
        #print(arr1_R[i_position])
        label='reff/a99=' + str(arr_reff_HSTS[i_position])[:-12]
        plt.plot(hs_thomson['Time'], np.abs(arr_dTedR[:,i_position]), 'ro', label=label)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        plt.xlim(4.398787, 4.403755)
        plt.ylim(0, 20)
        #plt.ylim(0,1)
        plt.title('SN' + self.ShotNo, loc='right')
        plt.xlabel('Time [sec]', fontsize=16)
        plt.ylabel('dTe/dR [keV/m]')
        plt.legend()
        plt.show()
        '''

        '''
        #fig = plt.figure(figsize=(3,1.5), dpi=150)
        fig = plt.figure(figsize=(5,2), dpi=150)
        #plt.errorbar(arr_reff_HSTS, arr_all[2, :], yerr=arr_all_dTe[2, :], capsize=5, fmt='o')
        #plt.errorbar(arr_reff_HSTS, arr_all[21, :], yerr=arr_all_dTe[21, :],capsize=5,  fmt='o')
        i_time = 2
        #print(arr1_R[i_position])
        label='t=' + str(np.array(hs_thomson['Time'][i_time])) + 's'
        plt.fill_between(arr_reff_HSTS, arr_all[i_time, :]-arr_all_dTe[i_time, :], arr_all[i_time, :]+arr_all_dTe[i_time, :], alpha=0.5, color='green', edgecolor='white')
        plt.plot(arr_reff_HSTS, arr_all[i_time, :], '-', label=label, linewidth=1, color='green')
        arr_reff_data_err = np.vstack([np.vstack([arr_reff_HSTS, arr_all[i_time, :]]), arr_all_dTe[i_time, :]]).T
        filename = 'HSTS_reff_Te_dTe_SN' +  self.ShotNo+ label + ".txt"
        np.savetxt(filename, arr_reff_data_err)
        i_time = 21
        label='t=' + str(np.array(hs_thomson['Time'][i_time])) + 's'
        plt.fill_between(arr_reff_HSTS, arr_all[i_time, :]-arr_all_dTe[i_time, :], arr_all[i_time, :]+arr_all_dTe[i_time, :], alpha=0.5, color='black', edgecolor='white')
        plt.plot(arr_reff_HSTS, arr_all[i_time, :], '-', label=label, linewidth=1, color='black')
        filename = 'HSTS_reff_Te_dTe_SN' +  self.ShotNo+ label + ".txt"
        arr_reff_data_err = np.vstack([np.vstack([arr_reff_HSTS, arr_all[i_time, :]]), arr_all_dTe[i_time, :]]).T
        np.savetxt(filename, arr_reff_data_err)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        #plt.plot(hs_thomson['Time'], arr_all[:, 13:15], '-')
        plt.legend(fontsize=8)
        plt.ylim(0, 6)
        #plt.xlim(0, 1.2)
        plt.xlim(0, 1.0)
        plt.xlabel('$r_{eff}/a_{99}$')
        plt.ylabel('$T_e [keV]$')
        plt.title('SN' + self.ShotNo, loc='right')
        plt.show()
        #plt.plot(hs_thomson['R'], hs_thomson['Te'].sel(Time=4.498876, method='nearest'))
        #plt.show()
        '''

        '''
        arr_dR_s = np.squeeze(arr_dR)
        arr_dTedR = np.abs(arr_dT/arr_dR_s)
        fig = plt.figure(figsize=(3,2), dpi=150)
        i_time =2
        label='t=' + str(np.array(hs_thomson['Time'][i_time])) + 's'
        plt.plot(np.abs(arr_reff_HSTS[1:]), arr_dTedR[i_time, :], 'ro', label=label, markersize=3)
        i_time = 21
        label='t=' + str(np.array(hs_thomson['Time'][i_time])) + 's'
        plt.plot(np.abs(arr_reff_HSTS[1:]), arr_dTedR[i_time, :], 'bo', label=label, markersize=3)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        #plt.plot(hs_thomson['Time'], arr_all[:, 13:15], '-')
        plt.legend(fontsize=7)
        plt.ylim(0, 35)
        plt.xlim(0, 1.2)
        plt.xlabel('$r_{eff}/a_{99}$')
        plt.ylabel('$dT_e/dR [keV/m]$')
        plt.title('SN' + self.ShotNo, loc='right')
        plt.show()
        #plt.plot(hs_thomson['R'], hs_thomson['Te'].sel(Time=4.498876, method='nearest'))
        #plt.show()
        '''

        '''
        thomson_slice = hs_thomson.sel(R=3.973)
        plt.plot(thomson_slice['Time'],thomson_slice['Te'])
        plt.show()

        thomson_slice_av = hs_thomson.sel(R=slice(3.802, 3.973))
        plt.plot(thomson_slice_av['Time'],thomson_slice_av['Te'].median(dim='R'))
        plt.show()
        '''

        '''
        arr_dR_s = np.squeeze(arr_dR)
        arr_dTedR = np.abs(arr_dT/arr_dR_s)
        plt.plot(hs_thomson['Time'], np.max(arr_dTedR[:, :], axis=1), 'ro')
        plt.plot(hs_thomson['Time'], np.abs(np.min(arr_dTedR[:, :], axis=1)), 'bo')
        #plt.xlim(3.3, 5.3)
        plt.show()
        '''

    def ana_plot_highSP_TS(self, arr_time):
        plt.clf()
        f, axarr = plt.subplots(3, 4, gridspec_kw={'wspace': 0, 'hspace': 0})

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        data = AnaData.retrieve('high_time_res_ts_model_fit', self.ShotNo, 1)
        #data = AnaData.retrieve('high_time_res_thomson', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        #t_mse = data_mse.getDimData('t')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        #reff_mse = data_mse.getValData('reff/a99')
        #Te = data.getValData('Te')
        #dTe = data.getValData('dTe')
        #ne = data.getValData('ne_calFIR')
        #dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        #iota = data_mse.getValData('iota')
        #iota_vac = data_mse.getValData('iota_vac')
        #iota_data_ip = data_mse.getValData('iota_data_ip')
        #iota_data_error_ip = data_mse.getValData('iota_data_error_ip')

        # 各時刻毎の R 対 Te のデータをプロット

        #plt.figure(figsize=(5.3, 3), dpi=100)
        #plt.xlabel('reff/a99')
        #plt.ylabel('iota, Te [keV], ne [e19m-3]')
        labels = []

        idx_arr_time = find_closest(t, arr_time)
        #idx_arr_time_mse = find_closest(t_mse, arr_time)
        val_arr_time = t[idx_arr_time]

        ax1 = plt.subplot(3,1,1)
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        plt.subplots_adjust(hspace=0.001, wspace=0)
        #label = 't = %05.3f (sec)' % (t[idx])  # 凡例用文字列
        title = ('tsmap_smooth #%d, ') % (self.ShotNo)
        #plt.title(title)
        # データ取得
        for i, idx in enumerate(idx_arr_time):
            r = reff[idx, :]
            #T = Te[idx, :]  # R 次元でのスライスを取得
            #dT = dTe[idx, :]  # R 次元でのスライスを取得
            #n = ne[idx, :]  # R 次元でのスライスを取得
            #dn = dne[idx, :]  # R 次元でのスライスを取得
            T_fit = Te_fit[idx, :]  # R 次元でのスライスを取得
            n_fit = ne_fit[idx, :]  # R 次元でのスライスを取得
            #plt.plot(r, T)
            #plt.plot(r, n)
            ax2.plot(r, T_fit, label='Te')
            ax1.plot(r, n_fit, label='ne')

        for i, idx in enumerate(idx_arr_time_mse):
            r_1d = reff_mse[idx, :]
            iota_1d = iota[idx, :]  # R 次元でのスライスを取得
            iota_vac_1d = iota_vac[idx, :]  # R 次元でのスライスを取得
            iota_data_ip_1d = iota_data_ip[idx, :]  # R 次元でのスライスを取得
            iota_data_error_ip_1d = iota_data_error_ip[idx, :]  # R 次元でのスライスを取得
            #plt.plot(r, T)
            #plt.plot(r, n)
            ax3.plot(r_1d, iota_1d, label='iota')
            #plt.plot(r_1d, iota_data_ip_1d, label='iota_data_ip')
            ax3.errorbar(r_1d, iota_data_ip_1d, yerr=iota_data_error_ip_1d, capsize=5, fmt='o', markersize=5)#, markeredgecolor='black', color='1')
            ax3.plot(r_1d, iota_vac_1d, label='iota_vac')

        #plt.legend()  # 凡例セット
        #plt.ylim(0, 5)
        #plt.xlim(0, 1.5)

        xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
        plt.setp(xticklabels, visible=False)

        plt.show()  # グラフ描画

    def ana_plot_eg(self, arr_time):
        plt.clf()
        f, axarr = plt.subplots(3, 4, gridspec_kw={'wspace': 0, 'hspace': 0})

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.show()
        plt.close()
        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)
        data_mse = AnaData.retrieve('lhdmse1_iota_ms', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('reff')
        dnum = data_mse.getDimNo()
        for i in range(dnum):
            print(i, data_mse.getDimName(i), data_mse.getDimUnit(i))
        vnum = data_mse.getValNo()
        for i in range(vnum):
            print(i, data_mse.getValName(i), data_mse.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        t_mse = data_mse.getDimData('t')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        reff_mse = data_mse.getValData('reff/a99')
        #Te = data.getValData('Te')
        #dTe = data.getValData('dTe')
        #ne = data.getValData('ne_calFIR')
        #dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        iota = data_mse.getValData('iota')
        iota_vac = data_mse.getValData('iota_vac')
        iota_data_ip = data_mse.getValData('iota_data_ip')
        iota_data_error_ip = data_mse.getValData('iota_data_error_ip')

        # 各時刻毎の R 対 Te のデータをプロット

        #plt.figure(figsize=(5.3, 3), dpi=100)
        #plt.xlabel('reff/a99')
        #plt.ylabel('iota, Te [keV], ne [e19m-3]')
        labels = []

        idx_arr_time = find_closest(t, arr_time)
        idx_arr_time_mse = find_closest(t_mse, arr_time)
        val_arr_time = t[idx_arr_time]

        ax1 = plt.subplot(3,1,1)
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        plt.subplots_adjust(hspace=0.001, wspace=0)
        #label = 't = %05.3f (sec)' % (t[idx])  # 凡例用文字列
        title = ('tsmap_smooth #%d, ') % (self.ShotNo)
        #plt.title(title)
        # データ取得
        for i, idx in enumerate(idx_arr_time):
            r = reff[idx, :]
            #T = Te[idx, :]  # R 次元でのスライスを取得
            #dT = dTe[idx, :]  # R 次元でのスライスを取得
            #n = ne[idx, :]  # R 次元でのスライスを取得
            #dn = dne[idx, :]  # R 次元でのスライスを取得
            T_fit = Te_fit[idx, :]  # R 次元でのスライスを取得
            n_fit = ne_fit[idx, :]  # R 次元でのスライスを取得
            #plt.plot(r, T)
            #plt.plot(r, n)
            ax2.plot(r, T_fit, label='Te')
            ax1.plot(r, n_fit, label='ne')

        for i, idx in enumerate(idx_arr_time_mse):
            r_1d = reff_mse[idx, :]
            iota_1d = iota[idx, :]  # R 次元でのスライスを取得
            iota_vac_1d = iota_vac[idx, :]  # R 次元でのスライスを取得
            iota_data_ip_1d = iota_data_ip[idx, :]  # R 次元でのスライスを取得
            iota_data_error_ip_1d = iota_data_error_ip[idx, :]  # R 次元でのスライスを取得
            #plt.plot(r, T)
            #plt.plot(r, n)
            ax3.plot(r_1d, iota_1d, label='iota')
            #plt.plot(r_1d, iota_data_ip_1d, label='iota_data_ip')
            ax3.errorbar(r_1d, iota_data_ip_1d, yerr=iota_data_error_ip_1d, capsize=5, fmt='o', markersize=5)#, markeredgecolor='black', color='1')
            ax3.plot(r_1d, iota_vac_1d, label='iota_vac')

        #plt.legend()  # 凡例セット
        #plt.ylim(0, 5)
        #plt.xlim(0, 1.5)

        xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
        plt.setp(xticklabels, visible=False)

        plt.show()  # グラフ描画

    def ana_delaytime_radh_allch(self, t_st, t_ed, isCondAV=None):
        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data_radhinfo.getValData('rho')
        fig = plt.figure(figsize=(8,5), dpi=150)
        arr_ydata = []
        arr_timedata_peaks_ltd = []
        arr_timedata_peaks_ltd_2 = []
        arr_tau_A_ltd = []
        arr_b_ltd = []
        arr_err_b_ltd = []
        arr_err_timedata_peaks_ltd = []
        if isCondAV:
            print("timedata and voltdata_array are loaded.")
            timedata, voltdata_array = self.combine_shots_data()
        for i_ch in range(32):
            try:
                print(self.ShotNo, i_ch)
                if isCondAV:
                    _,array_tau_A_ltd,array_b_ltd,timedata_peaks_ltd,err_timedata_peaks_ltd,array_err_b_ltd,timedata_peaks_ltd_2 = self.findpeaks_radh(t_st, t_ed, i_ch, timedata=timedata+1, voltdata_array=voltdata_array, isCondAV=True)
                    timedata_peaks_ltd -= 1
                else:
                    _,array_tau_A_ltd,_,timedata_peaks_ltd,err_timedata_peaks_ltd,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
                ydata = r[i_ch]*np.ones(len(timedata_peaks_ltd))
                #plt.plot(timedata_peaks_ltd, ydata, 's')
                #plt.plot(ydata, timedata_peaks_ltd, 'rs')
                #plt.plot(ydata, array_tau_A_ltd, 'rs')
                #plt.plot(ydata, array_b_ltd, 'rs')
                plt.plot(ydata, timedata_peaks_ltd_2-timedata_peaks_ltd, 'rs')
                arr_ydata.extend(ydata)
                arr_timedata_peaks_ltd.extend(timedata_peaks_ltd)
                arr_timedata_peaks_ltd_2.extend(timedata_peaks_ltd_2)
                arr_err_timedata_peaks_ltd.extend(err_timedata_peaks_ltd)
                arr_tau_A_ltd.extend(array_tau_A_ltd)
                arr_b_ltd.extend(array_b_ltd)
                arr_err_b_ltd.extend(array_err_b_ltd)
            except Exception as e:
                print(e)

        title = '%s, #%d' % (data_name_pxi, self.ShotNo)
        plt.title(title)
        plt.ylabel('Time [sec]')
        plt.xlabel('reff/a99')
        plt.xlim(0,1.2)
        #plt.ylim(-0.001, 0)
        #plt.ylim(4.4, 4.6)
        #plt.ylabel('R [m]')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        #filename = 'peaktime_%s_raw_tau_A_%d' % (data_name_pxi, self.ShotNo)
        filename = 'risetime_%s_raw_tau_A_%d_condAV' % (data_name_pxi, self.ShotNo)
        #plt.savefig(filename)
        #filename = 'risetime_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        #np.savetxt(filename, np.r_[arr_ydata, arr_timedata_peaks_ltd, arr_err_timedata_peaks_ltd])
        #filename = 'time_tau_errtau_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        #np.savetxt(filename, np.r_[arr_ydata, arr_tau_A_ltd, arr_err_timedata_peaks_ltd])
        filename_2 = 'r_b_berr_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        np.savetxt(filename_2, np.r_[arr_ydata, arr_b_ltd, arr_err_b_ltd])
        filename_3 = 'r_tpeak_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        np.savetxt(filename_3, np.r_[arr_ydata, arr_timedata_peaks_ltd_2])
        filename_4 = 'r_dtpeak_x0err_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        dtpeak = np.array(arr_timedata_peaks_ltd_2)-np.array(arr_timedata_peaks_ltd)
        np.savetxt(filename_4, np.r_[arr_ydata, dtpeak, arr_err_timedata_peaks_ltd])


    def find_peaks_radh_allch(self, t_st, t_ed):
        for i_ch in range(32):
            try:
                self.findpeaks_radh(t_st, t_ed, i_ch)
            except Exception as e:
                print(e)

    def findpeaks_radh(self, t_st, t_ed, i_ch, timedata=None, voltdata_array=None, isCondAV=None, isPlot=None, Mode_find_peaks=None, prom_min=None, prom_max=None):
        #prominences = np.array([])

        if isCondAV:
            print("timedata and voltdata_array are loaded.")
            timedata_ltd = timedata

        else:
            data_name_info = 'radhinfo' #radlinfo
            data_name_pxi = 'RADHPXI' #RADLPXI
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

            # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
            r = data_radhinfo.getValData('R')
            idx_time_st = find_closest(timedata, t_st)
            idx_time_ed = find_closest(timedata, t_ed)
            voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
            voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i_ch+1).get_val()
            voltdata_array[:, i_ch] = voltdata[idx_time_st:idx_time_ed]
            timedata_ltd = timedata[idx_time_st:idx_time_ed]
            #plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
            label = 'R=%.3fm' % r[i_ch]
        if Mode_find_peaks == "avalanche20240403":
            peaks, _ = find_peaks(voltdata_array[:, i_ch], prominence=(prom_min, prom_max), width=300, distance=500) #for avalanche 20240403
            prominences = peak_prominences(voltdata_array[:, i_ch], peaks)
            #print('prominences:', prominences)
        elif Mode_find_peaks == 'avalanche20240403_ece':
            peaks, properties = find_peaks(voltdata_array[:, i_ch], prominence=(prom_min, prom_max), width=60,
                                  distance=100)  # for avalanche 20240403
            prominences = peak_prominences(voltdata_array[:, i_ch], peaks)
            x = voltdata_array[:, i_ch]
            contour_heights = x[peaks] - prominences
            '''
            plt.plot(x)
            plt.plot(peaks, x[peaks], "x")
            plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
            plt.xlim(25000,45000)
            plt.show()
            '''
            '''
            fig = plt.figure(figsize=(8,3), dpi=150)
            plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            plt.plot(timedata_ltd, x)#, color='black')#, label='signal')
            plt.plot(timedata_ltd[peaks], x[peaks], "x", label='peak', color="red")
            #plt.vlines(x=timedata_ltd[peaks], ymin=contour_heights,
            #print(timedata_ltd[peaks])
            #print(peaks)
            #print(properties["prominences"])
            plt.vlines(x=timedata_ltd[peaks], ymin=x[peaks] - properties["prominences"],
                        ymax = x[peaks], color = "red", label = "prominence")
            # プロパティの値を整数に変換
            left_ips = np.round(properties["left_ips"]).astype(int)
            right_ips = np.round(properties["right_ips"]).astype(int)

            # 水平線を描画
            plt.hlines(y=properties["width_heights"], xmin=timedata_ltd[left_ips],
                       xmax=timedata_ltd[right_ips], color='green', label='width')
            plt.xlim(4.25,4.45)
            plt.ylim(0,1)
            plt.legend(fontsize=8, loc="upper right")
            plt.xlabel('Time[s]')
            plt.ylabel('Intensity of ECE [a.u.]')
            plt.show()
            plt.plot(x)
            plt.plot(peaks, x[peaks], "x")
            plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
                       ymax = x[peaks], color = "C1")
            plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                       xmax=properties["right_ips"], color = "C1")
            plt.xlim(25000,45000)
            plt.show()
            '''
        else:
            peaks, _ = find_peaks(voltdata_array[:, i_ch], prominence=0.001, width=40)     #default
        #peaks, _ = find_peaks(x, distance=5000)

        f, axarr = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8,8), dpi=150, sharex=True)
        data_name_pxi = 'RADHPXI'  # RADLPXI

        if isPlot:
            axarr[0].set_xlim(t_st, t_ed)
            axarr[0].plot(timedata_ltd, voltdata_array[:, i_ch])
            axarr[0].plot(timedata_ltd[peaks], voltdata_array[peaks, i_ch], "or")
            #if Mode_find_peaks == "avalanche20240403" or "avalanche20240403_ece":
            if Mode_find_peaks in ["avalanche20240403", "avalanche20240403_ece"]:
                title = '%s(ch.%d), Prominence:%.1fto%.1f, #%d' % (data_name_pxi, i_ch, prom_min, prom_max, self.ShotNo)
            else:
                title = '%s(ch.%d), #%d' % (data_name_pxi, i_ch, self.ShotNo)
            axarr[0].set_title(title, fontsize=16, loc='right')

        array_A = []
        array_tau = []
        array_err_tau = []
        array_tau_ltd = []
        array_err_tau_ltd = []
        array_b = []
        array_x0 = []
        array_err_x0 = []
        array_err_b = []
        array_timedata_peaks_ltd = []

        for i_peak in range(len(peaks)):
            width = 5000
            width_2 = 2000
            pre_peak = 200
            ydata = voltdata_array[peaks[i_peak]:peaks[i_peak]+width, i_ch]
            xdata = timedata_ltd[peaks[i_peak]:peaks[i_peak]+width]
            try:
                popt, pcov = curve_fit(func_exp_XOffset_wx0, xdata, ydata, bounds=([xdata[0], -np.inf, 0, 0], [xdata[1], np.inf, 1, 0.02]), p0=(xdata[0], -2.3, 0.2, 0.003))
                array_A.append(popt[2])
                array_tau.append(popt[3])
                perr_tau = np.sqrt(np.diag(pcov))
                array_err_tau.append(perr_tau[3])
            except Exception as e:
                print(e)
            #axarr[0].plot(xdata, func_exp_XOffset_wx0(xdata, *popt), 'r-')
            #print(popt)
            if isPlot:
                try:
                    ydata_2 = voltdata_array[peaks[i_peak]-width_2:peaks[i_peak]-pre_peak, i_ch]
                    xdata_2 = timedata_ltd[peaks[i_peak]-width_2:peaks[i_peak]-pre_peak]
                    popt_2, pcov_2 = curve_fit(func_LeakyReLU_like, xdata_2, ydata_2, bounds=([-np.inf, -np.inf, xdata_2[0], np.min(ydata_2)], [np.inf, np.inf, xdata_2[-1], np.max(ydata_2)]), p0=(0, 0.0001, xdata_2[-int(width_2/5)], ydata_2[0]))
                    #axarr[0].plot(xdata_2, func_LeakyReLU_like(xdata_2, *popt_2), 'y-')
                    #print(popt_2)
                    #plt.plot(xdata_2, ydata_2)
                    #plt.plot(xdata_2, func_LeakyReLU_like(xdata_2, *popt_2), 'g-')
                    plt.show()
                    array_b.append(popt_2[1])
                    array_x0.append(popt_2[2])
                except Exception as e:
                    print(e)

        timedata_peaks = timedata_ltd[peaks]
        idx_array = np.where((np.array(array_A) < 0.001) | (np.array(array_tau) > 0.019) | (np.array(array_tau) < 0.001))
        array_A_ltd = np.delete(array_A, idx_array)
        peaks_ltd = np.delete(peaks, idx_array)
        array_tau = np.delete(array_tau, idx_array)
        array_err_tau = np.delete(array_err_tau, idx_array)

        timedata_peaks_ltd = np.delete(timedata_peaks, idx_array)
        for i, i_peak_ltd in enumerate(peaks_ltd):
            width_2 = 2000
            pre_peak = 200
            try:
                ydata_2 = voltdata_array[i_peak_ltd-width_2:i_peak_ltd-pre_peak, i_ch]
                xdata_2 = timedata_ltd[i_peak_ltd-width_2:i_peak_ltd-pre_peak]
                popt_2, pcov_2 = curve_fit(func_LeakyReLU_like, xdata_2, ydata_2, bounds=([-np.inf, -np.inf, xdata_2[0], np.min(ydata_2)], [np.inf, np.inf, xdata_2[-1], np.max(ydata_2)]), p0=(0, 0.0001, xdata_2[-int(width_2/5)], ydata_2[0]))
                #if isCondAV:
                #    xdata_2 -= 1
                #    popt_2[2] -= 1
                #axarr[0].plot(xdata_2, func_LeakyReLU_like(xdata_2, *popt_2), 'y-')
                #print(popt_2)
                array_b.append(popt_2[1])
                array_x0.append(popt_2[2])
                perr = np.sqrt(np.diag(pcov_2))
                array_err_b.append(perr[1])
                array_err_x0.append(perr[2])
                array_tau_ltd.append(array_tau[i])
                array_err_tau_ltd.append(array_err_tau[i])
                array_timedata_peaks_ltd.append(timedata_peaks_ltd[i])
            except Exception as e:
                print(e)
        idx_array_2 = np.where(np.array(array_b) < 0)
        array_b_ltd = np.delete(array_b, idx_array_2)
        array_x0_ltd = np.delete(array_x0, idx_array_2)
        array_err_b_ltd = np.delete(array_err_b, idx_array_2)
        array_err_x0_ltd = np.delete(array_err_x0, idx_array_2)
        array_tau_ltd = np.delete(array_tau_ltd, idx_array_2)
        array_err_tau_ltd = np.delete(array_err_tau_ltd, idx_array_2)
        timedata_peaks_ltd_2 = np.delete(array_timedata_peaks_ltd, idx_array_2)

        #if isCondAV:
        #    timedata_ltd -= 1
        #    timedata_peaks_ltd -= 1

        if array_A_ltd.__len__() < 2 or array_b_ltd.__len__() < 2 or array_tau_ltd.__len__() < 2:
            mean_A = array_A_ltd
            mean_tau = array_tau_ltd
            mean_b = array_b_ltd
            print('Only 1 peak is found')
            #label_A = 'value: %.6f' % array_A_ltd
            #label_tau = 'value: %.6f\nerror: %.6f' % (array_tau_ltd, array_err_tau_ltd)
            #label_b = 'x0=%.6f\nb=%.4f' % (array_x0_ltd, array_b_ltd)
            label_A=''
            label_b=''
        else:
            mean_A = statistics.mean(array_A_ltd)
            mean_tau = statistics.mean(array_tau_ltd)
            mean_b = statistics.mean(array_b_ltd)
            stdev_A = statistics.stdev(array_A_ltd)
            stdev_tau = statistics.stdev(array_tau_ltd)
            stdev_b = statistics.stdev(array_b_ltd)
            label_A = 'mean:%.6f, stdev:%.6f' % (mean_A, stdev_A)
            label_tau = 'mean:%.6f, stdev:%.6f' % (mean_tau, stdev_tau)
            label_b = 'mean:%.6f, stdev:%.6f' % (mean_b, stdev_b)

        #mean_A = statistics.mean(array_A_ltd)


        if isPlot:
            axarr[1].plot(array_x0_ltd, array_b_ltd, 'og', label=label_b)
            #axarr[1].plot(array_x0, array_b, 'og', label=label_b)
            axarr[1].legend(frameon=True, loc='upper left', fontsize=15)
            axarr[1].set_ylabel('b', fontsize=15)
            #axarr[1].set_ylim(0, 2e-2)
            #axarr[1].set_xlim(t_st, t_ed)
            axarr[0].set_xlim(np.min(timedata_ltd), np.max(timedata_ltd))
            axarr[1].set_xlim(np.min(timedata_ltd), np.max(timedata_ltd))
            axarr[2].set_xlim(np.min(timedata_ltd), np.max(timedata_ltd))
            axarr[3].set_xlim(np.min(timedata_ltd), np.max(timedata_ltd))
            #axarr[2].plot(timedata_peaks_ltd, array_tau_ltd, 'or', label=label_tau)
            axarr[2].legend(frameon=True, loc='upper left', fontsize=15)
            #plt.title('tau(0,0.02)')
            axarr[2].set_ylabel('tau(0,0.02)', fontsize=15)
            axarr[2].set_ylim(0, 2e-2)
            #axarr[2].set_xlim(t_st, t_ed)
            try:
                axarr[3].plot(timedata_peaks_ltd, array_A_ltd, 'ob', label=label_A)
            except Exception as e:
                print(e)
            axarr[3].legend(frameon=True, loc='lower left', fontsize=15)
            #plt.title('A(0,1)')
            #axarr[3].set_xlim(t_st, t_ed)
            axarr[3].set_xlabel('Time [sec]', fontsize=15)
            axarr[3].set_ylabel('A(0,1)', fontsize=15)
            axarr[3].set_ylim(0,)
            axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            if Mode_find_peaks in ["avalanche20240403", "avalanche20240403_ece"]:
            #if Mode_find_peaks == "avalanche20240403" or "avalanche20240403_ece":
                filename = '%s_ch%d_raw_tau_A_prom%.1fto%.1f_%d.png' % (data_name_pxi, i_ch, prom_min, prom_max,self.ShotNo)
            else:
                filename = '%s_ch%d_raw_tau_A_pro01_04_%d' % (data_name_pxi, i_ch, self.ShotNo)
            #plt.savefig(filename)
            plt.show()
            plt.close()

        '''
        if isPlot:
            plt.plot(xdata, ydata, 'b-')
            try:
                plt.plot(xdata, func_exp_XOffset(xdata, *popt), 'r-', label=label)
            except Exception as e:
                print(e)
            plt.legend()
            plt.show()
        '''
        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

        print("%d peaks were found" % len(peaks))

        #if Mode_find_peaks == "avalanche20240403" or "avalanche20240403_ece":
        if Mode_find_peaks in ["avalanche20240403", "avalanche20240403_ece"]:
            return peaks, prominences, timedata_peaks_ltd_2#array_err_x0_ltd#timedata_peaks_ltd
        else:
            return array_A_ltd, array_tau_ltd, array_b_ltd, array_x0_ltd, array_err_tau_ltd, array_err_b_ltd, timedata_peaks_ltd_2#array_err_x0_ltd#timedata_peaks_ltd

    def findpeaks_mp(self, t_st, t_ed, i_ch, timedata=None, voltdata_array=None, isCondAV=None, isPlot=True):
        if isCondAV:
            print("timedata and voltdata_array are loaded.")
            timedata_ltd = timedata

        else:
            data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
            data_mp_2d = data_mp.getValData(0)
            timedata = data_mp.getDimData('TIME')

            data_name_info = 'radhinfo' #radlinfo

            # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
            idx_time_st = find_closest(timedata, t_st)
            idx_time_ed = find_closest(timedata, t_ed)
            timedata_ltd = timedata[idx_time_st:idx_time_ed]
        #peaks, _ = find_peaks(data_mp_2d[idx_time_st:idx_time_ed, i_ch], prominence=0.001, width=40)
        data_mp = data_mp_2d[idx_time_st:idx_time_ed, i_ch]
        data_mp_env = np.abs(signal.hilbert(data_mp))
        #peaks, _ = find_peaks(data_mp_env, prominence=3.5)  #175132
        #peaks, _ = find_peaks(data_mp_env, prominence=1, distance=1000) #169702
        #peaks, _ = find_peaks(data_mp_env, prominence=1, distance=100) #175346
        peaks, _ = find_peaks(data_mp_env, prominence=1, distance=500) #20211214-20211216, 20220113
        #peaks, _ = find_peaks(data_mp_env, prominence=1e-1, distance=300) #20211029

        timedata_peaks = timedata_ltd[peaks]
        if isPlot:
            f, axarr = plt.subplots(1, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8,4), dpi=150, sharex=True)

            axarr.set_xlim(t_st, t_ed)
            axarr.plot(timedata_ltd, data_mp)
            axarr.plot(timedata_ltd, data_mp_env)
            axarr.plot(timedata_ltd[peaks], data_mp_env[peaks], "or")
            title = '%s(ch.%d), #%d' % ('mp_raw_p', i_ch, self.ShotNo)
            axarr.set_title(title, fontsize=18, loc='right')


            axarr.set_xlim(np.min(timedata_ltd), np.max(timedata_ltd))
            axarr.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr.set_xlabel('Time [sec]', fontsize=15)
            #filename = '%s_ch%d_raw_tau_A_%d' % (data_name_pxi, i_ch, self.ShotNo)
            #plt.savefig(filename)
            plt.show()

        '''
        plt.plot(xdata, ydata, 'b-')
        plt.plot(xdata, func_exp_XOffset(xdata, *popt), 'r-', label=label)
        plt.legend()
        plt.show()
        '''
        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

        return timedata_peaks

    def condAV_DBS_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None, data_name_DBS=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        t = data.getDimData('Time')

        #data_name_DBS = 'mwrm_9o_comb_Iamp'
        #data_name_DBS = 'mwrm_comb_R_power'
        #data_name_DBS = 'mwrm_9o_R_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        #data_comb_R_power = AnaData.retrieve(data_name_comb_R_power, self.ShotNo, 1)
        #data_9o_R_Iamp = AnaData.retrieve(data_name_9o_R_Iamp, self.ShotNo, 1)

        t_data_DBS = data_DBS.getDimData("Time")
        #t_data_9o_comb_Iamp = data_9o_comb_Iamp.getDimData("Time")
        #t_data_comb_R_power = data_comb_R_power.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(t_data_DBS), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))


        #r = data_radhinfo.getValData('rho')
        #r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))


        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                                   timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_DBS =t_data_DBS[find_closest(t_data_DBS, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_data_DBS, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

                buf_DBS = data_DBS_2D[find_closest(t_data_DBS, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_data_DBS, timedata_peaks_ltd[i]+t_ed_condAV), :]

            else:
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_DBS = t_data_DBS[find_closest( t_data_DBS, timedata_peaks_ltd[i] - t_st_condAV):find_closest( t_data_DBS,
                                                                                                                        timedata_peaks_ltd[
                                                                                                                            i] + t_ed_condAV)] - \
                                timedata_peaks_ltd[i]
                    buf_DBS += data_DBS_2D[find_closest( t_data_DBS, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_data_DBS, timedata_peaks_ltd[i]+t_ed_condAV), :]
                except Exception as e:
                    print(e)

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_DBS = buf_DBS/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        label_pci = data_name_DBS
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        #a = ax2.plot(buf_t_DBS, cond_av_DBS, 'g-', label=label_pci, zorder=2)
        a = ax2.plot(buf_t_DBS, cond_av_DBS, '-', zorder=2)
        plt.title(data_name_DBS + ', Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('DBS', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()



        return buf_t_radh, cond_av_radh, buf_t_DBS, cond_av_DBS, rho, len(timedata_peaks_ltd), fs

    def condAV_BSADC_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None, data_name_BS=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        t = data.getDimData('Time')

        #ch1(4.0m)
        #ch_I = 17
        #ch_Q = 18
        #ch2(4.4m)
        #ch_I = 23
        #ch_Q = 24
        #ch3
        ch_I = 19
        ch_Q = 20
        #ch4(140G,4.4m)
        #ch_I = 21
        #ch_Q = 22
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_BS, start_time)
        end_index = np.searchsorted(timedata_BS, end_time)
        filtered_time = timedata_BS[start_index:end_index]
        filtered_signal = voltdata_BS_I[start_index:end_index]

        # フーリエ変換
        fft_result = fft(filtered_signal)

        # 複素振幅の計算
        complex_amplitude = ifft(fft_result)

        # 複素振幅の絶対値を計算（複素振幅強度）
        amplitude_strength = np.abs(complex_amplitude)

        # バターワースバンドパスフィルタの設計
        #lowcut = 90000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        #lowcut = 5000#1000  # カットオフ周波数 1kHz
        #highcut = 30000#10000  # カットオフ周波数 10kHz
        lowcut = 50000#1000  # カットオフ周波数 1kHz
        highcut = 100000#10000  # カットオフ周波数 10kHz
        #lowcut = 1000  # カットオフ周波数 1kHz
        #highcut = 10000  # カットオフ周波数 10kHz
        #lowcut = 500  # カットオフ周波数 1kHz
        #lowcut = 1000  # カットオフ周波数 1kHz
        #highcut = 2700  # カットオフ周波数 10kHz
        #highcut = 10000  # カットオフ周波数 10kHz
        order = 2#3#5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        #b, a = butter(order, low, btype='low', analog=False)

        # 複素振幅強度にフィルタを適用
        filtered_amplitude_strength = np.abs(filtfilt(b, a, amplitude_strength))

        plt.plot(filtered_time, filtered_amplitude_strength)
        plt.show()


        #r = data_radhinfo.getValData('rho')
        #r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))


        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                                   timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_BS =filtered_time[find_closest(filtered_time, timedata_peaks_ltd[i]-t_st_condAV):find_closest(filtered_time, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

                buf_BS = filtered_amplitude_strength[find_closest(filtered_time, timedata_peaks_ltd[i]-t_st_condAV):find_closest(filtered_time, timedata_peaks_ltd[i]+t_ed_condAV)]

            else:
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_BS =filtered_time[find_closest(filtered_time, timedata_peaks_ltd[i] - t_st_condAV):find_closest(filtered_time,
                                                                                                                      timedata_peaks_ltd[
                                                                                                                          i] + t_ed_condAV)] - \
                                timedata_peaks_ltd[i]
                    buf_BS +=filtered_amplitude_strength[find_closest(filtered_time, timedata_peaks_ltd[i]-t_st_condAV):find_closest(filtered_time, timedata_peaks_ltd[i]+t_ed_condAV)]
                except Exception as e:
                    print(e)

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_BS = buf_BS/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        label_pci = data_name_BS
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        #a = ax2.plot(buf_t_BS, cond_av_BS, 'g-', label=label_pci, zorder=2)
        a = ax2.plot(buf_t_BS, cond_av_BS, '-', zorder=2)
        plt.title(data_name_BS + ', Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('BS', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()



        return buf_t_radh, cond_av_radh, buf_t_BS, cond_av_BS, rho, len(timedata_peaks_ltd), fs

    def condAV_wp_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None,
                                              prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188796
        for i in range(shot_st, shot_ed + 1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh, t_wp, buf_cond_av_wp, rho_radh, num_av, fs = self.condAV_wp_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh_ac = buf_cond_av_radh - np.mean(buf_cond_av_radh[-1000:, :], axis=0)
                    buf_cond_av_radh *= num_av
                    buf_cond_av_radh_ac *= num_av
                    buf_cond_av_wp *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_, t_wp, buf_cond_av_wp_, rho_radh, num_av, fs = self.condAV_wp_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_radh_ -= np.mean(buf_cond_av_radh_[-1000:, :], axis=0)
                    buf_cond_av_radh_ac += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_wp += (buf_cond_av_wp_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh = buf_cond_av_radh / buf_num_av
        cond_av_radh_ac = buf_cond_av_radh_ac / buf_num_av
        cond_av_wp = buf_cond_av_wp / buf_num_av

        i_ch = 16
        # label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        #b = ax2.plot(t_DBS, cond_av_DBS, 'g-', label='DBS', zorder=1)
        b = ax2.plot(t_wp, cond_av_wp, '-', zorder=1, label="Wp")
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        plt.title('Wp,Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('Wp [kJ]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        # ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        # plt.savefig("timetrace_condAV_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.savefig("timetrace_condAV_ece_radh_wp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

    def condAV_wp_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        t = data.getDimData('Time')

        data_wp = AnaData.retrieve('wp', self.ShotNo, 1)
        data_wp_ = data_wp.getValData('Wp')
        t_data_wp = data_wp.getDimData('Time')

        vnum = data_wp.getValNo()

        dnum = data_wp.getDimNo()
        for i in range(dnum):
            print(i, data_wp.getDimName(i), data_wp.getDimUnit(i))
        vnum = data_wp.getValNo()
        for i in range(vnum):
            print(i, data_wp.getValName(i), data_wp.getValUnit(i))


        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))


        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                                   timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_wp =t_data_wp[find_closest(t_data_wp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_data_wp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

                buf_wp = data_wp_[find_closest(t_data_wp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_data_wp, timedata_peaks_ltd[i]+t_ed_condAV)]

            else:
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                if Mode_ece_radhpxi_calThom:
                    buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                else:
                    buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                buf_t_wp = t_data_wp[find_closest( t_data_wp, timedata_peaks_ltd[i] - t_st_condAV):find_closest( t_data_wp,
                                                                                                                 timedata_peaks_ltd[
                                                                                                                     i] + t_ed_condAV)] - \
                           timedata_peaks_ltd[i]
                buf_wp += data_wp_[find_closest( t_data_wp, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_data_wp, timedata_peaks_ltd[i]+t_ed_condAV)]
                '''
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_wp = t_data_wp[find_closest( t_data_wp, timedata_peaks_ltd[i] - t_st_condAV):find_closest( t_data_wp,
                                                                                                                        timedata_peaks_ltd[
                                                                                                                            i] + t_ed_condAV)] - \
                                timedata_peaks_ltd[i]
                    buf_wp += data_wp_[find_closest( t_data_wp, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_data_wp, timedata_peaks_ltd[i]+t_ed_condAV)]
                except Exception as e:
                    print(e)
                '''

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_wp = buf_wp/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        #a = ax2.plot(buf_t_DBS, cond_av_DBS, 'g-', label=label_pci, zorder=2)
        a = ax2.plot(buf_t_wp, cond_av_wp, '-', zorder=2)
        plt.title('Wp, Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('wp [kJ]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()



        return buf_t_radh, cond_av_radh, buf_t_wp, cond_av_wp, rho, len(timedata_peaks_ltd), fs
    def condAV_pci_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        t = data.getDimData('Time')

        data_fast = AnaData.retrieve('pci_ch4_fft_fast', self.ShotNo, 1)

        dnum = data_fast.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data_fast.getValNo()
        for i in range(vnum):
            print(i, data_fast.getValName(i), data_fast.getValUnit(i))
        t_pci_fast = data_fast.getDimData('Time')
        Sxx_50to150kHz = data_fast.getValData('50-150(kHz)')
        Sxx_150to500kHz = data_fast.getValData('150-500(kHz)')
        Sxx_50to500kHz = data_fast.getValData('50-500(kHz)')


        #r = data_radhinfo.getValData('rho')
        #r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))


        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                                   timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_pci = t_pci_fast[find_closest(t_pci_fast, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_pci_fast, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

                buf_pci = Sxx_50to500kHz[find_closest(t_pci_fast, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_pci_fast, timedata_peaks_ltd[i]+t_ed_condAV)]

            else:
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_pci = t_pci_fast[find_closest(t_pci_fast, timedata_peaks_ltd[i] - t_st_condAV):find_closest(t_pci_fast,
                                                                                                      timedata_peaks_ltd[
                                                                                                          i] + t_ed_condAV)] - \
                                  timedata_peaks_ltd[i]
                    buf_pci += Sxx_50to500kHz[find_closest(t_pci_fast, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_pci_fast, timedata_peaks_ltd[i]+t_ed_condAV)]
                except Exception as e:
                    print(e)

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_pci = buf_pci/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        label_pci = "pci_ch4_fft_fast"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        a = ax2.plot(buf_t_pci, cond_av_pci, 'g-', label=label_pci, zorder=2)
        plt.title('Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('PCI', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()



        return buf_t_radh, cond_av_radh, buf_t_pci, cond_av_pci, rho, len(timedata_peaks_ltd), fs

    def condAV_BSADC_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None,
                                              prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188796
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        for i in range(shot_st, shot_ed + 1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh, t_BS, buf_cond_av_BS, rho_radh, num_av, fs = self.condAV_BSADC_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True, data_name_BS=data_name_BS)
                    buf_cond_av_radh_ac = buf_cond_av_radh - np.mean(buf_cond_av_radh[-1000:, :], axis=0)
                    buf_cond_av_radh *= num_av
                    buf_cond_av_radh_ac *= num_av
                    buf_cond_av_BS *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_, t_BS, buf_cond_av_BS_, rho_radh, num_av, fs = self.condAV_BSADC_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True, data_name_BS=data_name_BS)
                    buf_cond_av_radh += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_radh_ -= np.mean(buf_cond_av_radh_[-1000:, :], axis=0)
                    buf_cond_av_radh_ac += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_BS += (buf_cond_av_BS_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh = buf_cond_av_radh / buf_num_av
        cond_av_radh_ac = buf_cond_av_radh_ac / buf_num_av
        cond_av_BS = buf_cond_av_BS / buf_num_av

        i_ch = 16
        # label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        #b = ax2.plot(t_BS, cond_av_BS, 'g-', label='BS', zorder=1)
        b = ax2.plot(t_BS, cond_av_BS, '-', zorder=1)
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        plt.title(data_name_BS + ',Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('BS', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        # ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        # plt.savefig("timetrace_condAV_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.savefig("timetrace_condAV_ece_radh_" + data_name_BS + "_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

    def condAV_DBS_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None,
                                              prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188796
        #data_name_DBS = 'mwrm_9o_comb_Iamp'
        #data_name_DBS = 'mwrm_comb_R_power'
        data_name_DBS = 'mwrm_9o_R_Iamp'
        for i in range(shot_st, shot_ed + 1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh, t_DBS, buf_cond_av_DBS, rho_radh, num_av, fs = self.condAV_DBS_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True, data_name_DBS=data_name_DBS)
                    buf_cond_av_radh_ac = buf_cond_av_radh - np.mean(buf_cond_av_radh[-1000:, :], axis=0)
                    buf_cond_av_radh *= num_av
                    buf_cond_av_radh_ac *= num_av
                    buf_cond_av_DBS *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_, t_DBS, buf_cond_av_DBS_, rho_radh, num_av, fs = self.condAV_DBS_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True, data_name_DBS=data_name_DBS)
                    buf_cond_av_radh += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_radh_ -= np.mean(buf_cond_av_radh_[-1000:, :], axis=0)
                    buf_cond_av_radh_ac += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_DBS += (buf_cond_av_DBS_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh = buf_cond_av_radh / buf_num_av
        cond_av_radh_ac = buf_cond_av_radh_ac / buf_num_av
        cond_av_DBS = buf_cond_av_DBS / buf_num_av

        i_ch = 16
        # label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        #b = ax2.plot(t_DBS, cond_av_DBS, 'g-', label='DBS', zorder=1)
        b = ax2.plot(t_DBS, cond_av_DBS, '-', zorder=1)
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        plt.title(data_name_DBS + ',Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('DBS', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        # ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        # plt.savefig("timetrace_condAV_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.savefig("timetrace_condAV_ece_radh_" + data_name_DBS + "_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

    def condAV_pci_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None,
                                              prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188781#188796
        for i in range(shot_st, shot_ed + 1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh, t_pci_fast, buf_cond_av_pci, rho_radh, num_av, fs = self.condAV_pci_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh_ac = buf_cond_av_radh - np.mean(buf_cond_av_radh[-1000:, :], axis=0)
                    buf_cond_av_radh *= num_av
                    buf_cond_av_radh_ac *= num_av
                    buf_cond_av_pci *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_, t_pci_fast, buf_cond_av_pci_, rho_radh, num_av, fs = self.condAV_pci_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_radh_ -= np.mean(buf_cond_av_radh_[-1000:, :], axis=0)
                    buf_cond_av_radh_ac += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_pci += (buf_cond_av_pci_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh = buf_cond_av_radh / buf_num_av
        cond_av_radh_ac = buf_cond_av_radh_ac / buf_num_av
        cond_av_pci = buf_cond_av_pci / buf_num_av

        i_ch = 16
        # label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        b = ax2.plot(t_pci_fast, cond_av_pci, 'g-', label='pci_ch4_fft_fast', zorder=1)
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        plt.title('Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('PCI', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        # ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        # plt.savefig("timetrace_condAV_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        # plt.savefig("timetrace_condAV_ece_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

    def ana_plot_DBS_fft(self):

        data_name_9o_comb_Iamp = 'mwrm_9o_comb_Iamp'
        data_name_comb_R_power = 'mwrm_comb_R_power'
        data_name_9o_R_Iamp = 'mwrm_9o_R_Iamp'
        data_9o_comb_Iamp = AnaData.retrieve(data_name_9o_comb_Iamp, self.ShotNo, 1)
        data_comb_R_power = AnaData.retrieve(data_name_comb_R_power, self.ShotNo, 1)
        data_9o_R_Iamp = AnaData.retrieve(data_name_9o_R_Iamp, self.ShotNo, 1)
        #data = AnaData.retrieve('pci_ch4_fft_fast', self.ShotNo, 1)

        t_data_9o_R_Iamp = data_9o_R_Iamp.getDimData("Time")
        t_data_9o_comb_Iamp = data_9o_comb_Iamp.getDimData("Time")
        t_data_comb_R_power = data_comb_R_power.getDimData("Time")
        vnum = data_9o_R_Iamp.getValNo()
        for i in range(vnum):
            Amp_data_9o_R_Iamp = data_9o_R_Iamp.getValData(i)
            plt.plot(t_data_9o_R_Iamp, Amp_data_9o_R_Iamp)
        plt.xlim(4,6)
        plt.show()

        vnum = data_comb_R_power.getValNo()
        for i in range(vnum):
            Amp_data_comb_R_power = data_comb_R_power.getValData(i)
            plt.plot(t_data_comb_R_power, Amp_data_comb_R_power)
        plt.xlim(4,6)
        plt.show()

        vnum = data_9o_comb_Iamp.getValNo()
        for i in range(vnum):
            Amp_data_9o_comb_Iamp = data_9o_comb_Iamp.getValData(i)
            plt.plot(t_data_9o_comb_Iamp, Amp_data_9o_comb_Iamp)
        plt.xlim(4,6)
        plt.show()


        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        print(data_name_9o_R_Iamp)
        dnum = data_9o_R_Iamp.getDimNo()
        for i in range(dnum):
            print(i, data_9o_R_Iamp.getDimName(i), data_9o_R_Iamp.getDimUnit(i))
        vnum = data_9o_R_Iamp.getValNo()
        for i in range(vnum):
            print(i, data_9o_R_Iamp.getValName(i), data_9o_R_Iamp.getValUnit(i))

        print(data_name_comb_R_power)
        dnum = data_comb_R_power.getDimNo()
        for i in range(dnum):
            print(i, data_comb_R_power.getDimName(i), data_comb_R_power.getDimUnit(i))
        vnum = data_comb_R_power.getValNo()
        for i in range(vnum):
            print(i, data_comb_R_power.getValName(i), data_comb_R_power.getValUnit(i))

        print(data_name_9o_comb_Iamp)
        dnum = data_9o_comb_Iamp.getDimNo()
        for i in range(dnum):
            print(i, data_9o_comb_Iamp.getDimName(i), data_9o_comb_Iamp.getDimUnit(i))
        vnum = data_9o_comb_Iamp.getValNo()
        for i in range(vnum):
            print(i, data_9o_comb_Iamp.getValName(i), data_9o_comb_Iamp.getValUnit(i))


    def ana_plot_pci_fft(self):

        #sig = VoltData.retrieve('pci_raw_fft', self.ShotNo, 1, 1)
        data = AnaData.retrieve('pci_ch4_fft', self.ShotNo, 1)
        data_fast = AnaData.retrieve('pci_ch4_fft_fast', self.ShotNo, 1)
        #data = AnaData.retrieve('pci_ch4_fft_fast', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        dnum = data_fast.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data_fast.getValNo()
        for i in range(vnum):
            print(i, data_fast.getValName(i), data_fast.getValUnit(i))
        t_pci = data.getDimData('Time')
        t_pci_fast = data_fast.getDimData('Time')
        Freq = data.getDimData('Freq')
        Sxx = data.getValData('Sxx')
        Sxx_50to150kHz = data_fast.getValData('50-150(kHz)')
        Sxx_150to500kHz = data_fast.getValData('150-500(kHz)')
        Sxx_50to500kHz = data_fast.getValData('50-500(kHz)')
        plt.pcolormesh(t_pci, Freq, Sxx.T, cmap='jet', vmin=0, vmax=0.001)
        plt.xlim(4.44, 4.55)
        cb = plt.colorbar()
        plt.show()
        print('test')
        plt.plot(t_pci, np.mean(Sxx, axis=1), 'r-')
        plt.plot(t_pci_fast, Sxx_50to150kHz, '-')
        plt.plot(t_pci_fast, Sxx_150to500kHz, 'b-')
        plt.plot(t_pci_fast, Sxx_50to500kHz, 'g-')
        plt.show()

    def ana_plot_pci_test(self):
        data = AnaData.retrieve('ece_fast', self.ShotNo, 1)
        r = data.getDimData('R')
        vnum = data.getValNo()
        t_ece = data.getDimData('Time')
        Te = data.getValData('Te')

        data = AnaData.retrieve('pci_profs', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')
        t = data.getDimData('time')
        reff_a99 = data.getDimData('reff/a99')
        ntilde_pci = data.getValData(0)
        v_pci = data.getValData(1)
        cosv_pci = data.getValData(2)
        #fig = plt.figure(figsize=(12,6), dpi=150)
        #plt.pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet')
        #plt.xlim(4.0, 4.65)
        #plt.show()

        data_pci_sdens_rvt = AnaData.retrieve('pci_sdens_rvt', self.ShotNo, 1)
        data_pci_sdens_rkt = AnaData.retrieve('pci_sdens_rkt', self.ShotNo, 1)
        t_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('time')
        t_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('time')
        v_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('v')
        k_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('k')
        reff_a99_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('reff/a99')
        reff_a99_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('reff/a99')
        sdens_rvt = data_pci_sdens_rvt.getValData(0)
        sdens_rkt = data_pci_sdens_rkt.getValData(0)
        #i_time=1380
        images = []
        for i_time in range(2000, 2050):
        #for i_time in range(10):
            fig, axarr = plt.subplots(5, 1, gridspec_kw={'wspace': 0, 'hspace': 0.25, 'height_ratios': [4,4,1,1,1]}, figsize=(8, 11), dpi=150,
                                      sharex=False)
            # pp.set_clim(1e1, 1e3)
            axarr[1].set_xlabel('reff/a99', fontsize=10)
            axarr[0].set_ylabel('k [1/mm]', fontsize=10)
            axarr[1].set_ylabel('v [km/s]', fontsize=10)
            axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            #fig = plt.figure()
            #ax1 = fig.add_subplot(211)
            axarr[0].set_title('#%d, t=%.3fsec' % (self.ShotNo, t_pci_sdens_rkt[i_time]), fontsize=20, loc='left')
            mappable0 = axarr[0].pcolormesh(reff_a99_pci_sdens_rkt, k_pci_sdens_rkt, sdens_rkt[i_time].T, cmap='jet', norm=LogNorm(vmin=1e3, vmax=1e5))
            mappable1 = axarr[1].pcolormesh(reff_a99_pci_sdens_rvt, v_pci_sdens_rvt, sdens_rvt[i_time].T, cmap='jet', norm=LogNorm(vmin=1e-2, vmax=1e0))
            pp0 = fig.colorbar(mappable0, ax=axarr[0], orientation="vertical", pad=0.01)
            pp1 = fig.colorbar(mappable1, ax=axarr[1], orientation="vertical", pad=0.01)
            pp0.set_label('PSD', fontname="Arial", fontsize=10)
            pp1.set_label('PSD', fontname="Arial", fontsize=10)
            #plt.show()

            axarr[2].set_ylabel('reff/a99', fontsize=10)
            axarr[3].set_ylabel('reff/a99', fontsize=10)
            axarr[4].set_ylabel('Te [keV]', fontsize=10)
            axarr[4].set_xlabel('Time [sec]', fontsize=10)
            axarr[3].set_xlim(3.0, 5.00)
            axarr[2].set_xlim(3.0, 5.00)
            axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            im0 = axarr[2].pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet',
                                      norm=LogNorm(vmin=1e6, vmax=1e7))
            im1 = axarr[3].pcolormesh(t, reff_a99, v_pci.T, cmap='jet')
            #im2 = axarr[2].pcolormesh(t, reff_a99, cosv_pci.T, cmap='jet', vmin=-0.5, vmax=0.5)
            pp0 = fig.colorbar(im0, ax=axarr[2], pad=0.01)
            pp1 = fig.colorbar(im1, ax=axarr[3], pad=0.01)
            pp0.set_label('ntilde[a.u.]', fontname="Arial", fontsize=10)
            pp1.set_label('v [km/sec]', fontname="Arial", fontsize=10)

            #arr_erace = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 32, 33, 44,
            #             45, 46, 47, 50, 51, 58]
            arr_erace = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
            axarr[4].plot(t_ece, Te[:, arr_erace])
            axarr[4].set_ylim(0, 5)
            axarr[4].set_xlim(3.0, 5.398)
            axarr[4].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[2].axvline(t_pci_sdens_rkt[i_time], color='red')
            axarr[3].axvline(t_pci_sdens_rkt[i_time], color='red')
            axarr[4].axvline(t_pci_sdens_rkt[i_time], color='red')
            # plt.savefig("fmp_psd_No%d.png" % self.ShotNo)
            filename = 'pci_k_v_reff_time_%d_t%.4f.png' % (self.ShotNo, t_pci_sdens_rkt[i_time])
            plt.savefig(filename)
            plt.close()

    def ana_plot_pci(self):
        data_pci_sdens_rvt = AnaData.retrieve('pci_sdens_rvt', self.ShotNo, 1)
        data_pci_sdens_rkt = AnaData.retrieve('pci_sdens_rkt', self.ShotNo, 1)
        t_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('time')
        t_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('time')
        v_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('v')
        k_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('k')
        reff_a99_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('reff/a99')
        reff_a99_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('reff/a99')
        sdens_rvt = data_pci_sdens_rvt.getValData(0)
        sdens_rkt = data_pci_sdens_rkt.getValData(0)
        #i_time=1380
        images = []
        for i_time in range(440,650):
            fig, axarr = plt.subplots(2, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 9), dpi=150,
                                      sharex=True)
            # pp.set_clim(1e1, 1e3)
            axarr[1].set_xlabel('reff/a99', fontsize=10)
            axarr[0].set_ylabel('k [1/mm]', fontsize=10)
            axarr[1].set_ylabel('v [km/s]', fontsize=10)
            axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            #fig = plt.figure()
            #ax1 = fig.add_subplot(211)
            axarr[0].set_title('#%d, t=%.3fsec' % (self.ShotNo, t_pci_sdens_rkt[i_time]), fontsize=20, loc='left')
            mappable0 = axarr[0].pcolormesh(reff_a99_pci_sdens_rkt, k_pci_sdens_rkt, sdens_rkt[i_time].T, cmap='jet', norm=LogNorm(vmin=1e2, vmax=1e5))
            mappable1 = axarr[1].pcolormesh(reff_a99_pci_sdens_rvt, v_pci_sdens_rvt, sdens_rvt[i_time].T, cmap='jet', norm=LogNorm(vmin=1e-3, vmax=1e0))
            pp0 = fig.colorbar(mappable0, ax=axarr[0], orientation="vertical", pad=0.01)
            pp1 = fig.colorbar(mappable1, ax=axarr[1], orientation="vertical", pad=0.01)
            pp0.set_label('PSD', fontname="Arial", fontsize=10)
            pp1.set_label('PSD', fontname="Arial", fontsize=10)
            #plt.show()
            filename = 'pci_k_v_reff_%d_t%.4f.png' % (self.ShotNo, t_pci_sdens_rkt[i_time])
            plt.savefig(filename)
            plt.close()
            #im = Image.open(filename)
            #images.append(im)


        '''
        data = AnaData.retrieve('pci_profs', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')
        t = data.getDimData('time')
        reff_a99 = data.getDimData('reff/a99')
        ntilde_pci = data.getValData(0)
        v_pci = data.getValData(1)
        cosv_pci = data.getValData(2)
        #fig = plt.figure(figsize=(12,6), dpi=150)
        #plt.pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet')
        #plt.xlim(4.0, 4.65)
        #plt.show()

        fig, axarr = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(20,10), dpi=150, sharex=True)
        axarr[0].set_title("#%d" % self.ShotNo, fontsize=20, loc='left')
        axarr[0].set_ylabel('reff/a99', fontsize=10)
        axarr[1].set_ylabel('reff/a99', fontsize=10)
        axarr[2].set_ylabel('reff/a99', fontsize=10)
        axarr[2].set_xlabel('Time [sec]', fontsize=12)
        axarr[0].set_xlim(4.44, 4.65)
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        #axarr[0].set_ylim(0, 50)
        #axarr[1].set_ylim(0, 50)
        #axarr[2].set_ylim(0, 50)
        #axarr[0].set_xticklabels([])
        #axarr[1].set_xticklabels([])
        #im0 = axarr[0].pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet', vmin=0, vmax=0.8e6)
        im0 = axarr[0].pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet',
                                        norm=LogNorm(vmin=1e5, vmax=5e6))
        im1 = axarr[1].pcolormesh(t, reff_a99, v_pci.T, cmap='jet')
        im2 = axarr[2].pcolormesh(t, reff_a99, cosv_pci.T, cmap='jet', vmin=-0.5, vmax=0.5)
        pp0 = fig.colorbar(im0, ax=axarr[0], pad=0.01)
        pp1 = fig.colorbar(im1, ax=axarr[1], pad=0.01)
        pp2 = fig.colorbar(im2, ax=axarr[2], pad=0.01)
        pp0.set_label('ntilde[a.u.]', fontname="Arial", fontsize=10)
        pp1.set_label('v [km/sec]', fontname="Arial", fontsize=10)
        pp2.set_label('cosv', fontname="Arial", fontsize=10)
        #plt.savefig("fmp_psd_No%d.png" % self.ShotNo)
        plt.show()
        '''

    def conditional_average_highK_mppk(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None):
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power1' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        vnum = data.getValNo()
        t = data.getDimData('Time')

        data_highK = data.getValData('Amplitude (20-200kHz)')
        reff_a99_highK = data.getValData('reff/a99')

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        r = data_radhinfo.getValData('rho')

        i_ch = 12#14#13#9
        i_ch_mp = 0
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        voltdata_array = np.zeros((len(timedata), 32))
        for i in range(32):
            voltdata_array[:, i] = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()

        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)

        data_mp_2d = data_mp.getValData(0)
        t_mp = data_mp.getDimData('TIME')

        timedata_peaks_ltd = self.findpeaks_mp(t_st, t_ed, i_ch_mp, isPlot=False)

        arr_reff_a99_highK = []

        t_st_condAV = 2e-3
        t_ed_condAV = 10e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3), 0]
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                buf_mp_env = np.abs(signal.hilbert(buf_mp))
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                arr_reff_a99_highK.append(reff_a99_highK[find_closest(t, timedata_peaks_ltd[i])])

            else:
                try:
                    buf_radh += voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_highK += data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ -= np.average(buf_highK_[:100], axis=0)
                    #buf_highK += buf_highK_
                    buf_mp += data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                    buf_mp_env += np.abs(signal.hilbert(data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]))
                    buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    arr_reff_a99_highK.append(reff_a99_highK[find_closest(t, timedata_peaks_ltd[i])])
                except Exception as e:
                    print(e)

            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.set_axisbelow(True)
            ax2.set_axisbelow(True)
            plt.rcParams['axes.axisbelow'] = True
            b = ax2.plot(buf_t_mp, buf_mp, 'g-', label='mp_raw', zorder=1)
            label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
            c = ax1.plot( buf_t_radh, buf_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
            a = ax2.plot(buf_t_highK, 1e3*buf_highK, 'b-', label='highK', zorder=3)
            plt.title('#%d' % self.ShotNo, loc='right')
            ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
            ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            #ax1.legend((a, b, *c), ('highK', 'mp_raw', 'radh'), loc='upper left')
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
            ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
            fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            plt.show()
            '''

        mean_reff_a99_highK = statistics.mean(arr_reff_a99_highK)
        stdev_reff_a99_highK = statistics.stdev(arr_reff_a99_highK)

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_highK = buf_highK/len(timedata_peaks_ltd)
        cond_av_mp = buf_mp/len(timedata_peaks_ltd)
        cond_av_mp_env = buf_mp_env/len(timedata_peaks_ltd)


        val_prominence_highK = 0.001
        num_peaks = 0
        #peaks, _ = find_peaks(cond_av_highK, prominence=0.00003)#, distance=50) #20211214-20211216, 20220113
        #peaks, _ = find_peaks(cond_av_highK, prominence=val_prominence_highK)#, distance=50) #20211214-20211216, 20220113
        while(num_peaks < 1 and val_prominence_highK > 0):
            peaks, _ = find_peaks(cond_av_highK, prominence=val_prominence_highK)
            #print('prominence=%.5f' % val_prominence_highK )
            #print('num_peaks=%d' % len(peaks))
            val_prominence_highK -= 0.000005
            num_peaks = len(peaks)

        array_rtime_popt_perr_highK = np.zeros((5, 17))
        array_rtime_popt_perr_mp = np.zeros((5, 15))

        array_rtime_popt_perr_highK[:, 0] = self.ShotNo
        array_rtime_popt_perr_mp[:, 0] = self.ShotNo
        array_rtime_popt_perr_highK[:, 1] = t_st
        array_rtime_popt_perr_highK[:, 2] = t_ed
        array_rtime_popt_perr_mp[:, 1] = t_st
        array_rtime_popt_perr_mp[:, 2] = t_ed
        array_rtime_popt_perr_highK[:, 3] = mean_reff_a99_highK
        array_rtime_popt_perr_highK[:, 4] = stdev_reff_a99_highK


        #popt_1, pcov = curve_fit(laplace_asymmetric_pdf, 1e7*buf_t_highK, cond_av_highK, p0=(cond_av_highK[0], 1, 0, cond_av_highK[find_closest(buf_t_highK, 0)], 1e-1))
        popt_1, pcov = curve_fit(laplace_asymmetric_pdf, buf_t_highK, cond_av_highK, p0=(cond_av_highK[0], 1, 0, 1e4, 1e-8))
        perr_1 = np.sqrt(np.diag(pcov))
        array_rtime_popt_perr_highK[0, 5:7] = [buf_t_highK[0], buf_t_highK[-1]]
        array_rtime_popt_perr_highK[0, 7:12] = popt_1
        array_rtime_popt_perr_highK[0, 12:] = perr_1
        popt_2, pcov = curve_fit(laplace_asymmetric_pdf, buf_t_highK[:400] , cond_av_highK[:400], p0=(cond_av_highK[0], 1, 0, 1e4, 1e-7))
        perr_2 = np.sqrt(np.diag(pcov))
        array_rtime_popt_perr_highK[1, 5:7] = [buf_t_highK[0], buf_t_highK[400]]
        array_rtime_popt_perr_highK[1, 7:12] = popt_2
        array_rtime_popt_perr_highK[1, 12:] = perr_2
        popt_3, pcov = curve_fit(laplace_asymmetric_pdf, buf_t_highK[100:300] , cond_av_highK[100:300], p0=(cond_av_highK[0], 1, 0, 1e4, 1e-7))
        perr_3 = np.sqrt(np.diag(pcov))
        array_rtime_popt_perr_highK[2, 5:7] = [buf_t_highK[100], buf_t_highK[300]]
        array_rtime_popt_perr_highK[2, 7:12] = popt_3
        array_rtime_popt_perr_highK[2, 12:] = perr_3
        popt_4, pcov = curve_fit(laplace_asymmetric_pdf, buf_t_highK[150:250] , cond_av_highK[150:250], p0=(cond_av_highK[0], 1, 0, 1e4, 1e-7))
        perr_4 = np.sqrt(np.diag(pcov))
        array_rtime_popt_perr_highK[3, 5:7] = [buf_t_highK[150], buf_t_highK[250]]
        array_rtime_popt_perr_highK[3, 7:12] = popt_4
        array_rtime_popt_perr_highK[3, 12:] = perr_4
        popt_5, pcov = curve_fit(laplace_asymmetric_pdf, buf_t_highK[170:230] , cond_av_highK[170:230], p0=(cond_av_highK[0], 1, 0, 1e4, 1e-7))
        perr_5 = np.sqrt(np.diag(pcov))
        array_rtime_popt_perr_highK[4, 5:7] = [buf_t_highK[170], buf_t_highK[230]]
        array_rtime_popt_perr_highK[4, 7:12] = popt_5
        array_rtime_popt_perr_highK[4, 12:] = perr_5

        #popt_mp_1, pcov_mp = curve_fit(laplace_asymmetric_pdf, 1e3*buf_t_mp, cond_av_mp_env, p0=(cond_av_mp_env[0], 1, 0, cond_av_mp_env[find_closest(buf_t_mp, 0)], 1e-2))
        popt_mp_1, pcov_mp = curve_fit(laplace_asymmetric_pdf, buf_t_mp, cond_av_mp_env, p0=(cond_av_mp_env[0], 1, 0, 1e4, 1e-4))
        perr_mp_1 = np.sqrt(np.diag(pcov_mp))
        array_rtime_popt_perr_mp[0, 3:5] = [buf_t_mp[0], buf_t_mp[-1]]
        array_rtime_popt_perr_mp[0, 5:10] = popt_mp_1
        array_rtime_popt_perr_mp[0, 10:] = perr_mp_1
        popt_mp_2, pcov_mp = curve_fit(laplace_asymmetric_pdf, buf_t_mp[:400], cond_av_mp_env[:400], p0=(cond_av_mp_env[0], 1, 0, 1e4, 1e-4))
        perr_mp_2 = np.sqrt(np.diag(pcov_mp))
        array_rtime_popt_perr_mp[1, 3:5] = [buf_t_mp[0], buf_t_mp[400]]
        array_rtime_popt_perr_mp[1, 5:10] = popt_mp_2
        array_rtime_popt_perr_mp[1, 10:] = perr_mp_2
        popt_mp_3, pcov_mp = curve_fit(laplace_asymmetric_pdf, buf_t_mp[100:300], cond_av_mp_env[100:300], p0=(cond_av_mp_env[0], 1, 0, 1e4, 1e-4))
        perr_mp_3 = np.sqrt(np.diag(pcov_mp))
        array_rtime_popt_perr_mp[2, 3:5] = [buf_t_mp[100], buf_t_mp[300]]
        array_rtime_popt_perr_mp[2, 5:10] = popt_mp_3
        array_rtime_popt_perr_mp[2, 10:] = perr_mp_3
        popt_mp_4, pcov_mp = curve_fit(laplace_asymmetric_pdf, buf_t_mp[150:250], cond_av_mp_env[150:250], p0=(cond_av_mp_env[0], 1, 0, 1e4, 1e-5))
        perr_mp_4 = np.sqrt(np.diag(pcov_mp))
        array_rtime_popt_perr_mp[3, 3:5] = [buf_t_mp[150], buf_t_mp[250]]
        array_rtime_popt_perr_mp[3, 5:10] = popt_mp_4
        array_rtime_popt_perr_mp[3, 10:] = perr_mp_4
        popt_mp_5, pcov_mp = curve_fit(laplace_asymmetric_pdf, buf_t_mp[170:230], cond_av_mp_env[170:230], p0=(cond_av_mp_env[0], 1, 0, 1e4, 1e-5))
        perr_mp_5 = np.sqrt(np.diag(pcov_mp))
        array_rtime_popt_perr_mp[4, 3:5] = [buf_t_mp[170], buf_t_mp[230]]
        array_rtime_popt_perr_mp[4, 5:10] = popt_mp_5
        array_rtime_popt_perr_mp[4, 10:] = perr_mp_5

        filename_popt_perr_highK = f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{self.ShotNo}'
        filename_popt_perr_mp = f'SN_tst_ted_rtime_popt_perr_mp_{self.ShotNo}'
        np.savetxt(filename_popt_perr_highK+'.txt', array_rtime_popt_perr_highK, delimiter=",")
        np.savetxt(filename_popt_perr_mp+'.txt', array_rtime_popt_perr_mp, delimiter=",")
        np.savez(filename_popt_perr_highK+'.npz', array_rtime_popt_perr_highK)
        np.savez(filename_popt_perr_mp+'.npz', array_rtime_popt_perr_mp)

        #popt, pcov = curve_fit(laplace_asymmetric_pdf, buf_t_highK, cond_av_highK, p0=(1, 0, cond_av_highK[200]))
        print(f'Snot No.:{self.ShotNo}')
        print(f'highK({buf_t_highK[0]:.2e}-{buf_t_highK[-1]:.2e}s): y0={popt_1[0]:.6e}, kappa={popt_1[1]:.6e}, loc={popt_1[2]:.6e}s, scale={popt_1[3]:.6e}, b={popt_1[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_1[1]**4)/(popt_1[3]**2*popt_1[1]**2)):.6e}s')
        print(perr_1)
        print(f'highK({buf_t_highK[0]:.2e}-{buf_t_highK[400]:.2e}s): y0={popt_2[0]:.6e}, kappa={popt_2[1]:.6e}, loc={popt_2[2]:.6e}s, scale={popt_2[3]:.6e}, b={popt_2[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_2[1]**4)/(popt_2[3]**2*popt_2[1]**2)):.6e}s')
        print(perr_2)
        print(f'highK({buf_t_highK[100]:.2e}-{buf_t_highK[300]:.2e}s): y0={popt_3[0]:.6e}, kappa={popt_3[1]:.6e}, loc={popt_3[2]:.6e}s, scale={popt_3[3]:.6e}, b={popt_3[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_3[1]**4)/(popt_3[3]**2*popt_3[1]**2)):.6e}s')
        print(perr_3)
        print(f'highK({buf_t_highK[150]:.2e}-{buf_t_highK[250]:.2e}s): y0={popt_4[0]:.6e}, kappa={popt_4[1]:.6e}, loc={popt_4[2]:.6e}s, scale={popt_4[3]:.6e}, b={popt_4[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_4[1]**4)/(popt_4[3]**2*popt_4[1]**2)):.6e}s')
        print(perr_4)
        print(f'highK({buf_t_highK[170]:.2e}-{buf_t_highK[230]:.2e}s): y0={popt_5[0]:.6e}, kappa={popt_5[1]:.6e}, loc={popt_5[2]:.6e}s, scale={popt_5[3]:.6e}, b={popt_5[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_5[1]**4)/(popt_5[3]**2*popt_5[1]**2)):.6e}s')
        print(perr_5)


        print(f'MP({buf_t_mp[0]:.2e}-{buf_t_mp[-1]:.2e}s): y0={popt_mp_1[0]:.6e}, kappa={popt_mp_1[1]:.6e}, loc={popt_mp_1[2]:.6e}s, scale={popt_mp_1[3]:.6e}, b={popt_mp_1[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_mp_1[1]**4)/(popt_mp_1[3]**2*popt_mp_1[1]**2)):.6e}s')
        print(perr_mp_1)
        print(f'MP({buf_t_mp[0]:.2e}-{buf_t_mp[400]:.2e}s): y0={popt_mp_2[0]:.6e}, kappa={popt_mp_2[1]:.6e}, loc={popt_mp_2[2]:.6e}s, scale={popt_mp_2[3]:.6e}, b={popt_mp_2[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_mp_2[1]**4)/(popt_mp_2[3]**2*popt_mp_2[1]**2)):.6e}s')
        print(perr_mp_2)
        print(f'MP({buf_t_mp[100]:.2e}-{buf_t_mp[300]:.2e}s): y0={popt_mp_3[0]:.6e}, kappa={popt_mp_3[1]:.6e}, loc={popt_mp_3[2]:.6e}s, scale={popt_mp_3[3]:.6e}, b={popt_mp_3[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_mp_3[1]**4)/(popt_mp_3[3]**2*popt_mp_3[1]**2)):.6e}s')
        print(perr_mp_3)
        print(f'MP({buf_t_mp[150]:.2e}-{buf_t_mp[250]:.2e}s): y0={popt_mp_4[0]:.6e}, kappa={popt_mp_4[1]:.6e}, loc={popt_mp_4[2]:.6e}s, scale={popt_mp_4[3]:.6e}, b={popt_mp_4[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_mp_4[1]**4)/(popt_mp_4[3]**2*popt_mp_4[1]**2)):.6e}s')
        print(perr_mp_4)
        print(f'MP({buf_t_mp[170]:.2e}-{buf_t_mp[230]:.2e}s): y0={popt_mp_5[0]:.6e}, kappa={popt_mp_5[1]:.6e}, loc={popt_mp_5[2]:.6e}s, scale={popt_mp_5[3]:.6e}, b={popt_mp_5[4]:.6e}, FWHM={2*np.sqrt(2*np.log(2)*(1+popt_mp_5[1]**4)/(popt_mp_5[3]**2*popt_mp_5[1]**2)):.6e}s')
        print(perr_mp_5)

        #print(pcov)

        fig = plt.figure(figsize=(12,6), dpi=150)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        b = ax2.plot(buf_t_mp, cond_av_mp, 'g-', label='mp_raw', zorder=1)
        b = ax2.plot(buf_t_mp, cond_av_mp_env, 'c-', label='mp_raw_env', zorder=1)
        b2 = ax2.plot(buf_t_mp, laplace_asymmetric_pdf(buf_t_mp, *popt_mp_1), label='mp-1')
        b2 = ax2.plot(buf_t_mp[:400], laplace_asymmetric_pdf(buf_t_mp[:400], *popt_mp_2), label='mp-2')
        b2 = ax2.plot(buf_t_mp[100:300], laplace_asymmetric_pdf(buf_t_mp[100:300], *popt_mp_3), label='mp-3')
        b2 = ax2.plot(buf_t_mp[150:250], laplace_asymmetric_pdf(buf_t_mp[150:250], *popt_mp_4), label='mp-4')
        b2 = ax2.plot(buf_t_mp[170:230], laplace_asymmetric_pdf(buf_t_mp[170:230], *popt_mp_5), label='mp-5')
        label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        a = ax2.plot(buf_t_highK, 1e3*cond_av_highK, 'b-', label='highK', zorder=3)
        a2 = ax2.plot(buf_t_highK[peaks], 1e3*cond_av_highK[peaks], "og")
        a3 = ax2.plot(buf_t_highK, 1e3*laplace_asymmetric_pdf(buf_t_highK, *popt_1), label='highK-1')
        a3 = ax2.plot(buf_t_highK[:400], 1e3*laplace_asymmetric_pdf(buf_t_highK[:400], *popt_2), label='highK-2')
        a3 = ax2.plot(buf_t_highK[100:300], 1e3*laplace_asymmetric_pdf(buf_t_highK[100:300], *popt_3), label='highK-3')
        a3 = ax2.plot(buf_t_highK[150:250], 1e3*laplace_asymmetric_pdf(buf_t_highK[150:250], *popt_4), label='highK-4')
        a3 = ax2.plot(buf_t_highK[170:230], 1e3*laplace_asymmetric_pdf(buf_t_highK[170:230], *popt_5), label='highK-5')
        if len(peaks) == 0:
            plt.title('reff/a99(highK)=%.4f+-%.4f, t_peak_highK=%.4fms(prominence=%.6f), %.2fs - %.2fs, conditional av.(%d points) #%d' % (mean_reff_a99_highK, stdev_reff_a99_highK, 1e3*buf_t_highK[peaks], val_prominence_highK, t_st, t_ed, len(timedata_peaks_ltd), self.ShotNo), loc='right')
        else:
            plt.title('reff/a99(highK)=%.4f+-%.4f, t_peak_highK=%.4fms(prominence=%.6f), %.2fs - %.2fs, conditional av.(%d points) #%d' % (mean_reff_a99_highK, stdev_reff_a99_highK, 1e3*buf_t_highK[peaks[0]], val_prominence_highK, t_st, t_ed, len(timedata_peaks_ltd), self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax1.set_xlabel('Time [sec]')
        ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        ax1.set_xlim(-0.002, 0.002)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        filename_popt_perr_highK = f'radh_highK_mp_laplace_{self.ShotNo}.png'
        plt.savefig(filename_popt_perr_highK)
        plt.show()


        return buf_highK, buf_mp, buf_radh, buf_t_highK, buf_t_mp, buf_t_radh, timedata_peaks_ltd, r, i_ch

    def combine_files(self, array_ShotNos, YYYYMMDD):
        #array_rtime_popt_perr_highK = np.zeros((5, 17))
        #array_rtime_popt_perr_mp = np.zeros((5, 15))
        array3D_highK = np.zeros((len(array_ShotNos), 5, 17))
        array3D_mp = np.zeros((len(array_ShotNos), 5, 15))
        for i in range(len(array_ShotNos)):
            filename_popt_perr_highK = f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{array_ShotNos[i]}.npz'
            filename_popt_perr_mp = f'SN_tst_ted_rtime_popt_perr_mp_{array_ShotNos[i]}.npz'
            array_rtime_popt_perr_highK = np.load(filename_popt_perr_highK)['arr_0']
            array_rtime_popt_perr_mp = np.load(filename_popt_perr_mp)['arr_0']
            array3D_highK[i, :, :] = array_rtime_popt_perr_highK
            array3D_mp[i, :, :] = array_rtime_popt_perr_mp

        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{YYYYMMDD}_1.txt', array3D_highK[:, 0, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_mp_{YYYYMMDD}_1.txt', array3D_mp[:, 0, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{YYYYMMDD}_2.txt', array3D_highK[:, 1, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_mp_{YYYYMMDD}_2.txt', array3D_mp[:, 1, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{YYYYMMDD}_3.txt', array3D_highK[:, 2, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_mp_{YYYYMMDD}_3.txt', array3D_mp[:, 2, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{YYYYMMDD}_4.txt', array3D_highK[:, 3, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_mp_{YYYYMMDD}_4.txt', array3D_mp[:, 3, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_highK_{YYYYMMDD}_5.txt', array3D_highK[:, 4, ], delimiter=',')
        np.savetxt(f'SN_tst_ted_reffa99_reffa99err_rtime_popt_perr_mp_{YYYYMMDD}_5.txt', array3D_mp[:, 4, ], delimiter=',')

    def condAV_radh_highK_mp_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188796
        for i in range(shot_st, shot_ed+1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh_01, t_highK, buf_cond_av_highK, t_mp, buf_cond_av_mp, rho_radh, num_av, fs = self.condAV_radh_highK_mp_trigLowPassRadh(t_st=t_st,
                                                                                                                                                                    t_ed=t_ed,
                                                                                                                                                                    prom_min=0.4,
                                                                                                                                                                    prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh_01 *= num_av
                    buf_cond_av_highK *= num_av
                    buf_cond_av_mp *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_01_, t_highK, buf_cond_av_highK_, t_mp, buf_cond_av_mp_, rho_radh, num_av, fs = self.condAV_radh_highK_mp_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh_01 += (buf_cond_av_radh_01_ * num_av)
                    buf_cond_av_highK += (buf_cond_av_highK_ * num_av)
                    buf_cond_av_mp += (buf_cond_av_mp_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh_01 = buf_cond_av_radh_01/buf_num_av
        cond_av_highK = buf_cond_av_highK/buf_num_av
        cond_av_mp = buf_cond_av_mp/buf_num_av

        fig = plt.figure(figsize=(10, 8), dpi=150)
        for i in range(len(rho_radh)):
            label_radh = 'r/a=%.2f' % rho_radh[i]
            plt.plot(t_radh, 0.05*cond_av_radh_01[:,i]-rho_radh[i])#, label=label_radh)
            #plt.plot(t_radh, cond_av_radh_01[:,i])#, label=label_radh)
            #plt.plot(t_radh, cond_av_radh_01[:,i], label=label_radh)
            plt.hlines(-rho_radh[i], t_radh[0], t_radh[-1], colors='black', linestyle='dashed', linewidth=0.2)
            #plt.legend()
        title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        plt.title(title, loc='right', fontsize=16)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        plt.xlim(t_radh[0], t_radh[-1])
        plt.ylim(-1, -0.1)
        plt.show()
        #plt.savefig("Teall_condAV_radh_LP%dkHz_No%dto%d_prom%.1fto%.1f.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("Te_ece_all_condAV_radh_LP%dkHz_No%dto%d_prom%.1fto%.1f.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.close()


        i_ch = 16
        #label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        b = ax2.plot(t_mp, cond_av_mp, 'g-', label='mp_raw', zorder=1)
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh_01[:, i_ch], 'r-', label=label_radh, zorder=2)
        #a = ax2.plot(t_highK, 1e3*cond_av_highK, 'b-', label=label_highK, zorder=3)
        a = ax2.plot(t_highK, 1e3*cond_av_highK, 'b-', zorder=3)
        plt.title('Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("timetrace_condAV_ece_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*t_radh, rho_radh, cond_av_radh_01.T, cmap='bwr', vmin=-0.05, vmax=0.05)
        cb = plt.colorbar()
        cb.set_label("dTe [keV]", fontsize=18)
        #cb.set_label("Te [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("condAV_radh_ece_LP%dkHz_No%dto%d_prom%.1fto%.1f_v4.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*t_radh[1:], rho_radh, (cond_av_radh_01[1:,:]-cond_av_radh_01[:-1,:]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        cb = plt.colorbar()
        cb.set_label("dTe/dt [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_dTedt_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("condAV_radh_ece_dTedt_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*t_radh, rho_radh[1:], (cond_av_radh_01[:,1:]-cond_av_radh_01[:,:-1]).T, cmap='bwr', vmin=-0.05, vmax=0.05)
        #plt.pcolormesh(1e3*t_radh, rho_radh[1:], (cond_av_radh_01[:,1:]-cond_av_radh_01[:,:-1]).T, cmap='jet_r', vmin=-0.1, vmax=0)
        cb = plt.colorbar()
        cb.set_label("dTe/dR [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_dTdR_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.savefig("condAV_radh_ece_dTdR_LP%dkHz_No%dto%d_prom%.1fto%.1f_v4.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

        return t_radh, rho_radh, cond_av_radh_01

    def condAV_radh_highK_mp_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        reff_a99_highK = data.getValData('reff/a99')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        t = data.getDimData('Time')

        data_highK = data.getValData('Amplitude (20-200kHz)')

        #r = data_radhinfo.getValData('rho')
        #r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))

        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_mp_2d = data_mp.getValData(0)
        t_mp = data_mp.getDimData('TIME')

        fs = 500#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3), 0]
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                '''
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_highK_ = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                buf_mp_ = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

            else:
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_highK_ = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    buf_mp_ = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_highK += data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ -= np.average(buf_highK_[:100], axis=0)
                    #buf_highK += buf_highK_
                    buf_mp += data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                except Exception as e:
                    print(e)
            '''
            try:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()
                ax1.set_axisbelow(True)
                ax2.set_axisbelow(True)
                plt.rcParams['axes.axisbelow'] = True
                b = ax2.plot(buf_t_mp, buf_mp_, 'g-', label='mp_raw', zorder=1)
                label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
                c = ax1.plot( buf_t_radh, buf_radh_[:, i_ch], 'r-', label=label_radh, zorder=2)
                a = ax2.plot(buf_t_highK, 1e3*buf_highK_, 'b-', label='highK', zorder=3)
                plt.title('#%d' % self.ShotNo, loc='right')
                ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
                ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                #ax1.legend((a, b, *c), ('highK', 'mp_raw', 'radh'), loc='upper left')
                ax2.set_xlabel('Time [sec]')
                ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
                ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
                fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
                ax1.set_zorder(2)
                ax2.set_zorder(1)
                ax1.patch.set_alpha(0)
                plt.show()
            except Exception as e:
                print(e)
            '''

        label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_highK = buf_highK/len(timedata_peaks_ltd)
        cond_av_mp = buf_mp/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        b = ax2.plot(buf_t_mp, cond_av_mp, 'g-', label='mp_raw', zorder=1)
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        a = ax2.plot(buf_t_highK, 1e3*cond_av_highK, 'b-', label=label_highK, zorder=3)
        plt.title('Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()

        cond_av_radh_01 = np.zeros((len(cond_av_radh[:,0]), 32))
        if Mode_ece_radhpxi_calThom:
            for i in range(32):
                cond_av_radh_01[:, i] = cond_av_radh[:, i]
                cond_av_radh_01[:, i] -= np.mean(cond_av_radh_01[-1000:,i])
        else:
            for i in range(32):
                cond_av_radh_01[:, i] = normalize_0_1(cond_av_radh[:, i])
                cond_av_radh_01[:, i] -= np.mean(cond_av_radh_01[-1000:,i])
        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dHz), Prominence:%.1fto%.1f, #%d' % (fs, prom_min, prom_max, self.ShotNo)
        plt.title(title, fontsize=16, loc='right')
        #plt.pcolormesh(1e3*buf_t_radh[:-1], rho, (cond_av_radh_01[1:,:]-cond_av_radh_01[:-1,:]).T, cmap='bwr', vmin=-0.001, vmax=0.001)
        #plt.pcolormesh(1e3*buf_t_radh, rho[:-1], (cond_av_radh_01[:,:-1]-cond_av_radh_01[:,1:]).T, cmap='bwr', vmin=-0.5, vmax=0.5)
        plt.pcolormesh(1e3*buf_t_radh, rho, cond_av_radh_01.T, cmap='jet')
        cb = plt.colorbar()
        cb.set_label("Te [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        #plt.show()
        plt.close()


        return buf_t_radh, cond_av_radh_01, buf_t_highK, cond_av_highK, buf_t_mp, cond_av_mp, rho, len(timedata_peaks_ltd), fs

    def conditional_average_highK(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None):
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        t = data.getDimData('Time')

        data_highK = data.getValData('Amplitude (20-200kHz)')

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        r = data_radhinfo.getValData('rho')

        i_ch = 13#20#14#13#9
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        voltdata_array = np.zeros((len(timedata), 32))
        for i in range(32):
            voltdata_array[:, i] = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()

        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_mp_2d = data_mp.getValData(0)
        t_mp = data_mp.getDimData('TIME')

        _, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch, isPlot=False)
        #_, _, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch)


        #timedata_peaks_ltd = timedata_peaks_ltd_
        #arr_peak_selection = [0,2,3,5,6,7]#169714
        #arr_peak_selection = [0,6,7,8,9,10,12,13]#169715
        #arr_peak_selection = [1,2,5,6,7,8]#169712
        arr_peak_selection =[0,1,2,3,4,10,11] #169710
        #arr_peak_selection =[2,3,4] #169692
        #arr_peak_selection =[1,2,8] #169693
        #arr_peak_selection =[5,7,10] #169698
        #arr_peak_selection = [10] #169699
        #arr_peak_selection = [3,4,10] #169707
        #arr_peak_selection = [3,4,5,8,10,11,13] #169708
        timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        arr_toff = [-2e-4, 0, -2e-4, -3e-4, -3e-4, -2e-4, -1e-4]    #169710
        #arr_toff = [-2e-4, -3.5e-4, -3.5e-4, -1.5e-4, -2.5e-4, 0.5e-4]    #169712
        #arr_toff = 1e-4*np.array([-3,-1,0,-0.5,-1,-3,-2,-0.5])    #169715
        #arr_toff = 1e-4*np.array([-1.2,0,-1,-0.5,-1,-1])    #169714
        #arr_toff = 1e-4*np.array([-1,-3.5,-4])    #169692
        #arr_toff = 1e-4*np.array([-3,-2,-2])    #169693
        #arr_toff = 1e-4*np.array([-2,-1,0])    #169698
        #arr_toff = 1e-4*np.array([-0.5])    #169699
        #arr_toff = 1e-4*np.array([1.2,-2.2,-1.0])    #169707
        #arr_toff = 1e-4*np.array([-0.6,-0.9,0.1,-1,-2,-3.2,0])    #169708
        #timedata_peaks_ltd = np.array(timedata_peaks_ltd, dtype=np.float64)
        #arr_toff = float(arr_toff)
        timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 2e-3
        t_ed_condAV = 4e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3), 0]
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

            else:
                try:
                    buf_radh += voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_highK += data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ -= np.average(buf_highK_[:100], axis=0)
                    #buf_highK += buf_highK_
                    buf_mp += data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                    buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                except Exception as e:
                    print(e)

            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.set_axisbelow(True)
            ax2.set_axisbelow(True)
            plt.rcParams['axes.axisbelow'] = True
            b = ax2.plot(buf_t_mp, buf_mp, 'g-', label='mp_raw', zorder=1)
            label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
            c = ax1.plot( buf_t_radh, buf_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
            a = ax2.plot(buf_t_highK, 1e3*buf_highK, 'b-', label='highK', zorder=3)
            plt.title('#%d' % self.ShotNo, loc='right')
            ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
            ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            #ax1.legend((a, b, *c), ('highK', 'mp_raw', 'radh'), loc='upper left')
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
            ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
            fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            plt.show()
        '''

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_highK = buf_highK/len(timedata_peaks_ltd)
        cond_av_mp = buf_mp/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        b = ax2.plot(buf_t_mp, cond_av_mp, 'g-', label='mp_raw', zorder=1)
        label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        a = ax2.plot(buf_t_highK, 1e3*cond_av_highK, 'b-', label='highK', zorder=3)
        plt.title('#%d' % self.ShotNo, loc='right')
        #ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', visible=True, which='major', color='#666666', linestyle='-')
        #ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax1.grid(axis='x', visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.show()


        return buf_highK, buf_mp, buf_radh, buf_t_highK, buf_t_mp, buf_t_radh, timedata_peaks_ltd, r, i_ch

    def combine_shots_data(self):
        arr_peak_selection =[0,1,2,3,4,10,11] #169710
        arr_toff = [-2e-4, 0, -2e-4, -3e-4, -3e-4, -2e-4, -1e-4]    #169710
        #arr_peak_selection = [0,2,3,5,6,7]#169714
        #arr_toff = 1e-4*np.array([-1.2,0,-1,-0.5,-1,-1])    #169714
        #arr_peak_selection =[2,3,4] #169692
        #arr_toff = 1e-4*np.array([-1,-3.5,-4])    #169692
        #arr_peak_selection =[5,7,10] #169698
        #arr_toff = 1e-4*np.array([-2,-1,0])    #169698
        #arr_peak_selection = [3,4,10] #169707
        #arr_toff = 1e-4*np.array([1.2,-2.2,-1.0])    #169707
        buf_highK, buf_mp, buf_radh, buf_t_highK, buf_t_mp, buf_t_radh, timedata_peaks_ltd, _, _ = self.conditional_average_highK(t_st=4.3, t_ed=4.6, arr_peak_selection=arr_peak_selection, arr_toff=arr_toff)
        arr_peak_selection = [1,2,5,6,7,8]#169712
        arr_toff = [-2e-4, -3.5e-4, -3.5e-4, -1.5e-4, -2.5e-4, 0.5e-4]    #169712
        #arr_peak_selection = [0,6,7,8,9,10,12,13]#169715
        #arr_toff = 1e-4*np.array([-3,-1,0,-0.5,-1,-3,-2,-0.5])    #169715
        #arr_peak_selection =[1,2]#,8] #169693
        #arr_toff = 1e-4*np.array([-3,-2])#,-2])    #169693
        #arr_peak_selection =[5,7,10] #169698
        #arr_toff = 1e-4*np.array([-2,-1,0])    #169698
        #arr_peak_selection = [10] #169699
        #arr_toff = 1e-4*np.array([-0.5])    #169699
        #arr_peak_selection = [3,4,5,8,10,11,13] #169708
        #arr_toff = 1e-4*np.array([-0.6,-0.9,0.1,-1,-2,-3.2,0])    #169708
        ShotNo_1 = self.ShotNo
        ShotNo_2 = self.ShotNo_2
        self.ShotNo = self.ShotNo_2
        buf_highK_2, buf_mp_2, buf_radh_2, buf_t_highK_2, buf_t_mp_2, buf_t_radh_2, timedata_peaks_ltd_2, r, i_ch = self.conditional_average_highK(t_st=4.3, t_ed=4.6, arr_peak_selection=arr_peak_selection, arr_toff=arr_toff)

        cond_av_radh = (buf_radh + buf_radh_2)/(len(timedata_peaks_ltd) + len(timedata_peaks_ltd_2))
        cond_av_highK = (buf_highK + buf_highK_2)/(len(timedata_peaks_ltd) + len(timedata_peaks_ltd_2))
        cond_av_mp = (buf_mp + buf_mp_2)/(len(timedata_peaks_ltd) + len(timedata_peaks_ltd_2))

        cond_av_radh -= np.average(cond_av_radh[:100, :], axis=0)

        fs = 2e4
        dt = buf_t_radh[1] - buf_t_radh[0]
        dt_highK = buf_t_highK[1] - buf_t_highK[0]
        dt_mp = buf_t_mp[1] - buf_t_mp[0]
        freq = fftpack.fftfreq(len( buf_t_radh), dt)
        freq_highK = fftpack.fftfreq(len(buf_t_highK), dt_highK)
        freq_mp = fftpack.fftfreq(len(buf_t_mp), dt_mp)
        '''
        low-pass filter for voltdata
        '''
        yf = fftpack.rfft(cond_av_radh[:, i_ch-4]) / (int(len(cond_av_radh[:, i_ch-4]) / 2))
        yf_highK = fftpack.rfft(cond_av_highK) / (int(len(cond_av_highK) / 2))
        yf_mp = fftpack.rfft(cond_av_mp) / (int(len(cond_av_mp) / 2))
        yf2 = np.copy(yf)
        yf2_highK = np.copy(yf_highK)
        yf2_mp = np.copy(yf_mp)
        yf2[(freq > fs)] = 0
        yf2_highK[(freq_highK > fs)] = 0
        yf2_mp[(freq_mp > fs)] = 0
        yf2[(freq < 0)] = 0
        yf2_highK[(freq_highK < 0)] = 0
        yf2_mp[(freq_mp < 0)] = 0
        y2 = np.real(fftpack.irfft(yf2) * (int(len(cond_av_radh[:, i_ch-4]) / 2)))
        y2_highK = np.real(fftpack.irfft(yf2_highK) * (int(len(cond_av_highK) / 2)))
        y2_mp = np.real(fftpack.irfft(yf2_mp) * (int(len(cond_av_mp) / 2)))

        ''
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        #b = ax2.plot(buf_t_mp, cond_av_mp, 'g-', label='mp_raw', zorder=1)
        b = ax2.plot(buf_t_mp, y2_mp, 'g-', label='mp_raw', zorder=1)
        label_radh = "ECE(rho=" + str(r[i_ch-4]) + ")"
        #c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch-4], 'r-', label=label_radh, zorder=2)
        c = ax1.plot(buf_t_radh, y2, 'r-', label=label_radh, zorder=2)
        ''
        '''
        label_radh = "ECE(rho=" + str(r[i_ch-8]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch-8], 'k-', label=label_radh, zorder=2)
        label_radh = "ECE(rho=" + str(r[i_ch-6]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch-6]+0.1, 'b-', label=label_radh, zorder=2)
        label_radh = "ECE(rho=" + str(r[i_ch-4]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch-4]+0.2, 'c-', label=label_radh, zorder=2)
        label_radh = "ECE(rho=" + str(r[i_ch-2]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch-2]+0.3, 'm-', label=label_radh, zorder=2)
        label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch]+0.4, 'r-', label=label_radh, zorder=2)
        label_radh = "ECE(rho=" + str(r[i_ch+2]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch+2]+0.5, 'g-', label=label_radh, zorder=2)
        label_radh = "ECE(rho=" + str(r[i_ch+4]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch+4]+0.6, 'y-', label=label_radh, zorder=2)
        '''
        d = ax2.plot(buf_t_highK, 1e3*cond_av_highK, 'c-', label='highK', zorder=3)
        a = ax2.plot(buf_t_highK, 1e3*y2_highK, 'b-', label='highK', zorder=3)
        plt.title('Low-pass(<%dHz), #%d, %d' % (fs, ShotNo_1, ShotNo_2), loc='right')
        ax1.grid(axis='x', visible=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax1.set_xlabel('Time [sec]')
        ax2.set_ylabel('Amplitude of e-scale turb.[a.u.]\nb_dot_tilda [T/s]', rotation=270, labelpad=15)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        ax1.set_xlim(np.min(buf_t_radh), np.max(buf_t_radh))
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.show()

        fig = plt.figure(figsize=(8,4), dpi=150)
        plt.title('#%d, %d' % (ShotNo_1, ShotNo_2), loc='right')
        plt.pcolormesh(r, buf_t_radh+0.0003, cond_av_radh, cmap='jet')
        cb = plt.colorbar(pad=0.01)
        cb.ax.tick_params(labelsize=15)
        cb.set_label("Intensity of ECE [a.u.]", fontsize=22, rotation=270, labelpad=25)
        plt.ylabel("Time [sec]", fontsize=22)
        plt.xlabel(r"$r_{eff}/a_{99}$", fontsize=22)
        plt.xlim(0, 1.2)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        plt.show()

        self.findpeaks_radh(t_st=4.3, t_ed=4.6, i_ch=14, timedata=buf_t_radh+1, voltdata_array=cond_av_radh, isCondAV=True)

        filename = '_%d_%d.txt' % (ShotNo_1, ShotNo_2)
        filename_filtered = '_LP%dHz_%d_%d.txt' % (fs ,ShotNo_1, ShotNo_2)
        filename_radh_filtered = '_ch%d_LP%dHz_%d_%d.txt' % (i_ch+1, fs ,ShotNo_1, ShotNo_2)
        #np.savetxt('radh_array_condAV' + filename, cond_av_radh)
        #np.savetxt('radh_timedata_condAV' + filename, buf_t_radh)
        np.savetxt('highK_array_condAV' + filename, cond_av_highK)
        np.savetxt('highK_timedata_condAV' + filename, buf_t_highK)
        #np.savetxt('mp_array_condAV' + filename, cond_av_mp)
        #np.savetxt('mp_timedata_condAV' + filename, buf_t_mp)
        np.savetxt('radh_array_condAV' + filename_radh_filtered, y2)
        np.savetxt('highK_array_condAV' + filename_filtered, y2_highK)
        #np.savetxt('mp_array_condAV' + filename_filtered, y2_mp)

        return buf_t_radh, cond_av_radh

    def fft_all_DBS(self, t_st=None, t_ed=None):
        #data_name_DBS = 'mwrm_9o_comb_Iamp'
        #data_name_DBS = 'mwrm_9o_R_Iamp'
        #data_name_DBS = 'mwrm_comb_R_Iamp'
        data_name_DBS = 'mwrm_comb_U_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1


        fig, ax = plt.subplots(figsize=(8,20), dpi=150)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        #i_ch = 14#13#9
        for i_ch in range(vnum):
            data = data_DBS.getValData(i)
            N = len(data)


            F = np.fft.fft(data)
            freq = np.fft.fftfreq(N, d=(timedata[1]-timedata[0]))

            Amp = np.abs(F)

            #ax.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)] )
            ax.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)]*10**i_ch)
        #plt.xlim(0, 3)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [#]")
        ax.set_title(data_name_DBS + ", #%d" % self.ShotNo)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(freq[1], freq[-1])
        plt.show()


    def fft_all_radh(self, t_st=None, t_ed=None):

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #radlpxi
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')
        # フィッティング結果を格納するリスト
        A_results = []
        A_errors = []
        pow_results = []
        pow_errors = []
        rho_results = []

        fig, ax = plt.subplots(figsize=(8,20), dpi=150)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        #i_ch = 14#13#9
        for i_ch in range(32):
            if i_ch == 0:
                timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
                idx_time_st = find_closest(timedata, t_st)
                idx_time_ed = find_closest(timedata, t_ed)

            voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i_ch])).get_val()
            N = len(voltdata[idx_time_st:idx_time_ed])


            F = np.fft.fft(voltdata[idx_time_st:idx_time_ed])
            freq = np.fft.fftfreq(N, d=(timedata[1]-timedata[0]))

            Amp = np.abs(F)

            # フィッティング範囲を変数として定義
            fit_range_min = 40  # フィッティング範囲の最小値
            fit_range_max = 400  # フィッティング範囲の最大値

            # フィッティング範囲のフィルタリング
            mask = (freq >= fit_range_min) & (freq <= fit_range_max)
            freq_filtered = freq[mask]
            Amp_filtered = Amp[mask]


            try:
                # フィッティングを実行
                popt, pcov = curve_fit(power_func, freq_filtered, Amp_filtered)
                # 共分散行列から標準誤差を計算
                perr = np.sqrt(np.diag(pcov))
                A_err, pow_err = perr

                # フィッティング結果の取得
                A_fit, pow_fit = popt
                print(f'Fitting result of Ch.{i_ch}(y=A*x^pow)(rho={rho[i_ch]}): A = {A_fit:.3f} ± {A_err:.3f}, pow = {pow_fit:.3f} ± {pow_err:.3f}')
                # フィッティング結果をリストに追加
                A_results.append(A_fit)
                A_errors.append(A_err)
                pow_results.append(pow_fit)
                pow_errors.append(pow_err)
                rho_results.append(rho[i_ch])

                '''
                # 元データとフィッティング結果をプロット（両対数グラフ）
                plt.figure(figsize=(10, 6))
                plt.loglog(freq, Amp, '-', label='Original Data')
                plt.loglog(freq_filtered, power_func(freq_filtered, *popt), '-',
                           label=f'Fitted Curve: y={A_fit:.3f}x^{pow_fit:.3f}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True, which="both", ls="--")
                plt.title(f'Power Law Fit in {fit_range_min}-{fit_range_max} Hz Range, Ch: {i_ch}')
                plt.show()
                '''

            except RuntimeError:
                # フィッティングが失敗した場合、Noneを追加
                A_results.append(None)
                A_errors.append(None)
                pow_results.append(None)
                pow_errors.append(None)
                rho_results.append(None)

        '''
            #ax.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)] )
            ax.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)]*10**i_ch)

        #plt.xlim(0, 3)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [#]")
        ax.set_title(data_name_pxi + ", #%d" % self.ShotNo)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(freq[1], freq[-1])
        plt.show()
        '''

        # フィッティング結果を配列に変換し、エラーのあるデータのみを選別
        A_results = np.array(A_results)
        A_errors = np.array(A_errors)
        pow_results = np.array(pow_results)
        pow_errors = np.array(pow_errors)
        rho_results = np.array(rho_results)

        # Noneを含むフィッティング結果を除外
        valid_mask = ~np.isnan(A_results) & ~np.isnan(A_errors) & ~np.isnan(pow_results) & ~np.isnan(pow_errors)
        A_results = A_results[valid_mask]
        A_errors = A_errors[valid_mask]
        pow_results = pow_results[valid_mask]
        pow_errors = pow_errors[valid_mask]
        rho_results = rho_results[valid_mask]

        '''
        # 結果をプロット（Aとpowのエラーバー付きグラフ）
        plt.figure(figsize=(12, 8))

        # Aの結果をプロット
        plt.subplot(2, 1, 1)
        plt.errorbar(rho_results, A_results, yerr=A_errors, fmt='o', label='A (Amplitude Factor)', color='b')
        plt.xlabel('rho')
        plt.ylabel('A (Amplitude Factor)')
        plt.title(f'Fitting Results for A, #{self.ShotNo}')
        plt.xlim(0,1)
        plt.grid(True)
        plt.legend()

        # powの結果をプロット
        plt.subplot(2, 1, 2)
        plt.errorbar(rho_results, pow_results, yerr=pow_errors, fmt='o', label='pow (Exponent)', color='r')
        plt.xlabel('rho')
        plt.ylabel('pow (Exponent)')
        plt.title(f'Fitting Results for pow, #{self.ShotNo}')
        plt.xlim(0,1)
        plt.grid(True)
        plt.legend()

        # グラフ表示
        plt.tight_layout()
        plt.savefig(f'fitting_power_radh_{self.ShotNo}.png')
        plt.show()
        '''
        # A_results, A_errors, pow_results, pow_errorsを2次元配列に並べる
        fit_results = np.column_stack((rho_results, A_results, A_errors, pow_results, pow_errors))

        # 2次元配列をテキストファイルに保存
        np.savetxt(f'fitting_results_rho_A_pow_{self.ShotNo}.txt', fit_results, header=f'rho_results_{self.ShotNo} A_results_{self.ShotNo} A_errors_{self.ShotNo} pow_results_{self.ShotNo} pow_errors_{self.ShotNo}', fmt='%.6f')





        '''
        i_ch = 16
        voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i_ch])).get_val()
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
        plt.show()
        plt.hist(voltdata, bins=100, edgecolor='black', density=True)
        plt.show()
        '''

    def cal_hurstd_radh(self, t_st=None, t_ed=None):

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'radhpxi' #radlpxi
        data_radhinfo = anadata.retrieve(data_name_info, self.shotno, 1)

        '''
        data_name_highk = 'mwrm_highk_power3' #radlpxi data_radhinfo = anadata.retrieve(data_name_info, self.shotno, 1)
        data = anadata.retrieve(data_name_highk, self.shotno, 1)
        vnum = data.getvalno()
        for i in range(vnum):
            print(i, data.getvalname(i), data.getvalunit(i))
        timedata = data.getdimdata('time')
        voltdata= data.getvaldata('amplitude (20-200khz)')
        voltdata += 1

        '''
        i_ch = 14#13#9
        timedata = timedata.retrieve(data_name_pxi, self.shotno, 1, 1).get_val()
        voltdata = voltdata.retrieve(data_name_pxi, self.shotno, 1, i_ch+1).get_val()

        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)

        plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
        plt.show()

        for lag in [100, 500, 1000, 5000, 10000, 50000]:# 300000]:
        #for lag in [30, 100, 300, 500, 2000, 10000, 30000, 50000, 100000]:# 300000]:
            hurst_exp = get_hurst_exponent(voltdata[idx_time_st:idx_time_ed], lag)
            print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")
            hurst_exp2 = hurst(voltdata[idx_time_st:idx_time_st+lag])
            print(f"Hurst exponent 2 with {lag} lags: {hurst_exp2:.4f}")

    def ana_plot_radh_highK_wfindpeaks(self, t_st=None, t_ed=None):
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power1' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power2' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        t = data.getDimData('Time')

        reff_a99_highK = data.getValData('reff/a99')
        #plt.plot(t, reff_a99_highK)
        #plt.show()
        data_highK = data.getValData('Amplitude (20-200kHz)')

        data_name_highK_probe = 'MWRM-PXI'
        voltdata_highK_probe = VoltData.retrieve(data_name_highK_probe,self.ShotNo, 1, 4).get_val()
        voltdata_highK_probe_01 = np.where(voltdata_highK_probe < -0.5, 1, 0)
        timedata_highK_probe = TimeData.retrieve(data_name_highK_probe, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        #data_name_info = 'radminfo' #radlinfo
        #data_name_pxi = 'RADMPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        r = data_radhinfo.getValData('rho')

        i_ch = 14#5#6#9#13#14
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i_ch+1).get_val()

        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_mp_2d = data_mp.getValData(0)
        t_mp = data_mp.getDimData('TIME')

        _, _, _, timedata_peaks_ltd,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        t_offset_st = 4.3915
        t_offset_ed = 4.3925
        offset_highK = np.mean(data_highK[find_closest(t, t_offset_st):find_closest(t, t_offset_ed)])
        data_highK -= offset_highK


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        arr_sum_highK = []
        arr_delta_rise_maxhighK = []
        arr_reff_a99_highK = []

        for i in range(len(timedata_peaks_ltd)):
            #d = ax1.axvspan(timedata_peaks_ltd[i]-2e-4, timedata_peaks_ltd[i]+8e-4, color='#ffcdd2')
            #d = ax1.vlines(timedata_peaks_ltd[i], -1,20, 'black')
            sum_highK = np.sum(data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)])
            time_max_highK = t[find_closest(t, timedata_peaks_ltd[i]-2e-4) + np.argmax(data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)])]
            #arg_max = np.argmax(data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)])
            #data_ltd = data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)]
            delta_rise_maxhighK = timedata_peaks_ltd[i] - time_max_highK
            arr_sum_highK.append(sum_highK)
            arr_delta_rise_maxhighK.append(delta_rise_maxhighK)
            arr_reff_a99_highK.append(reff_a99_highK[find_closest(t, timedata_peaks_ltd[i])])


        b = ax1.plot(t_mp, data_mp_2d[:,0], 'g-', label='mp_raw')
        #b = ax1.plot(timedata_highK_probe, voltdata_highK_probe, 'g-', label='highK_probe')
        b = ax1.plot(timedata_highK_probe, voltdata_highK_probe_01, '-', label='highK_probe_01')
        idx_time_st = find_closest(t, t_st)
        idx_time_ed = find_closest(t, t_ed)
        label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1))
        a = ax1.plot(t, 1000*data_highK, 'b-', label=label_highK)
        label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
        c = ax2.plot(timedata, voltdata, 'r-', label=label_radh)
        plt.title('#%d' % self.ShotNo, loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        #ax1.legend((a, b, *c), ('highK', 'mp_raw', 'radh'), loc='upper left')
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('b_dot_tilda [T/s]')
        ax2.set_ylabel('Intensity of ECE [a.u.]', rotation=270, labelpad=15)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.xlim(3.3, 5.3)
        #plt.xlim(4.40, 4.48)
        #plt.xlim(4.3, 4.50)
        #plt.xlim(4.398787, 4.403755)
        #ax1.set_ylim(-3,5)
        #ax2.set_ylim(-4.8,-3.3)
        #plt.savefig("timetrace_radh_highK_mp_t439to440_No%d.png" % self.ShotNo)
        plt.tight_layout()
        plt.show()

        mean_sum_highK = statistics.mean(arr_sum_highK)
        stdev_sum_highK = statistics.stdev(arr_sum_highK)
        mean_delta_rise_maxhighK = statistics.mean(arr_delta_rise_maxhighK)
        stdev_delta_rise_maxhighK = statistics.stdev(arr_delta_rise_maxhighK)
        mean_reff_a99_highK = statistics.mean(arr_reff_a99_highK)
        stdev_reff_a99_highK = statistics.stdev(arr_reff_a99_highK)

        print(mean_sum_highK, stdev_sum_highK)
        print(mean_delta_rise_maxhighK, stdev_delta_rise_maxhighK)
        print(mean_reff_a99_highK, stdev_reff_a99_highK)

        #plt.plot(arr_reff_a99_highK, arr_sum_highK, 'ro')
        #plt.show()

    def ana_plot_radh_highK(self, t_st=None, t_ed=None):
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power1' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power2' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        data_name_info = 'radhinfo' #radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        ch = data_radhinfo.getValData('pxi')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        t = data.getDimData('Time')

        reff_a99_highK = data.getValData('reff/a99')
        #plt.plot(t, reff_a99_highK)
        #plt.show()
        data_highK = data.getValData('Amplitude (20-200kHz)')

        data_name_highK_probe = 'MWRM-PXI'
        voltdata_highK_probe = VoltData.retrieve(data_name_highK_probe,self.ShotNo, 1, 4).get_val()
        voltdata_highK_probe_01 = np.where(voltdata_highK_probe < -0.5, 1, 0)
        timedata_highK_probe = TimeData.retrieve(data_name_highK_probe, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        #data_name_info = 'radminfo' #radlinfo
        #data_name_pxi = 'RADMPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        r = data_radhinfo.getValData('rho')

        i_ch = 16#11#5#6#9#13#14
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i_ch])).get_val()
        fs = 1e3
        dt = timedata[1] - timedata[0]
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        '''
        low-pass filter for voltdata
        '''
        yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
        yf2 = np.copy(yf)
        yf2[(freq > fs)] = 0
        yf2[(freq < 0)] = 0
        y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
        #voltdata_array[:, i] = y2
        #voltdata_array[:, i] = normalize_0_1(y2)

        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_mp_2d = data_mp.getValData(0)
        t_mp = data_mp.getDimData('TIME')

        #_, _, _, timedata_peaks_ltd,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        #t_offset_st = 5.3915#4.3915
        #t_offset_ed = 5.3925#4.3925
        t_offset_st = 4.3915
        t_offset_ed = 4.3925
        offset_highK = np.mean(data_highK[find_closest(t, t_offset_st):find_closest(t, t_offset_ed)])
        data_highK -= offset_highK


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        '''
        arr_sum_highK = []
        arr_delta_rise_maxhighK = []
        arr_reff_a99_highK = []

        for i in range(len(timedata_peaks_ltd)):
            #d = ax1.axvspan(timedata_peaks_ltd[i]-2e-4, timedata_peaks_ltd[i]+8e-4, color='#ffcdd2')
            #d = ax1.vlines(timedata_peaks_ltd[i], -1,20, 'black')
            sum_highK = np.sum(data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)])
            time_max_highK = t[find_closest(t, timedata_peaks_ltd[i]-2e-4) + np.argmax(data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)])]
            #arg_max = np.argmax(data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)])
            #data_ltd = data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-4):find_closest(t, timedata_peaks_ltd[i]+8e-4)]
            delta_rise_maxhighK = timedata_peaks_ltd[i] - time_max_highK
            arr_sum_highK.append(sum_highK)
            arr_delta_rise_maxhighK.append(delta_rise_maxhighK)
            arr_reff_a99_highK.append(reff_a99_highK[find_closest(t, timedata_peaks_ltd[i])])
        '''


        #b = ax1.plot(t_mp, data_mp_2d[:,0], 'g-', label='mp_raw')
        #b = ax1.plot(timedata_highK_probe, voltdata_highK_probe, 'g-', label='highK_probe')
        #b = ax1.plot(timedata_highK_probe, voltdata_highK_probe_01, '-', label='highK_probe_01')
        idx_time_st_highK = find_closest(t, t_st)
        idx_time_ed_highK = find_closest(t, t_ed)
        label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK[idx_time_st_highK:idx_time_ed_highK].mean(axis=-1))
        a = ax1.plot(t, 1000*data_highK+1.5, 'b-', label=label_highK)
        label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
        #c = ax2.plot(timedata, voltdata, 'r-', label=label_radh)
        c = ax2.plot(timedata[idx_time_st:idx_time_ed], normalize_0_1(y2), 'r-', label=label_radh)
        plt.title('#%d' % self.ShotNo, loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        #ax1.legend((a, b, *c), ('highK', 'mp_raw', 'radh'), loc='upper left')
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('b_dot_tilda [T/s]')
        ax2.set_ylabel('Intensity of ECE [a.u.]', rotation=270, labelpad=15)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        #plt.xlim(5.1, 5.2)
        #plt.xlim(4.40, 4.48)
        plt.xlim(5.0, 5.30)
        #plt.xlim(4.398787, 4.403755)
        #ax1.set_ylim(-3,5)
        ax2.set_ylim(0,2.5)
        #plt.savefig("timetrace_radh_highK_mp_t439to440_No%d.png" % self.ShotNo)
        plt.tight_layout()
        plt.show()

        '''
        mean_sum_highK = statistics.mean(arr_sum_highK)
        stdev_sum_highK = statistics.stdev(arr_sum_highK)
        mean_delta_rise_maxhighK = statistics.mean(arr_delta_rise_maxhighK)
        stdev_delta_rise_maxhighK = statistics.stdev(arr_delta_rise_maxhighK)
        mean_reff_a99_highK = statistics.mean(arr_reff_a99_highK)
        stdev_reff_a99_highK = statistics.stdev(arr_reff_a99_highK)

        print(mean_sum_highK, stdev_sum_highK)
        print(mean_delta_rise_maxhighK, stdev_delta_rise_maxhighK)
        print(mean_reff_a99_highK, stdev_reff_a99_highK)

        #plt.plot(arr_reff_a99_highK, arr_sum_highK, 'ro')
        #plt.show()
        '''


    def get_ne(self, target_t, target_r):
        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('reff')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))



        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff_a99 = data.getValData('reff/a99')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')



        idx_time = find_closest(t, target_t)
        r = reff_a99[idx_time, :]
        n = ne_fit[idx_time, :]  # R 次元でのスライスを取得
        idx_r = find_closest(r, target_r)
        plt.plot(r, n)
        plt.plot(r[idx_r], n[idx_r], 'ro')
        plt.xlabel('reff/a99')
        plt.ylabel('ne')
        title = '#%d' % (self.ShotNo)
        plt.title(title)
        plt.show()
        print(self.ShotNo)
        print(n[idx_r])

    def ana_plot_radh(self, t_st, t_ed):
        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_org = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        fs = 1e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            voltdata_array_org[:, i] = voltdata[idx_time_st:idx_time_ed]
            #plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
            label = 'R=%.3fm' % r[i]
            #plt.plot(timedata[idx_time_st:idx_time_ed], y2, label=label)
        #plt.xlim(4.4, 4.6)
        '''
        plt.legend()
        plt.xlabel('Time [sec]')
        title = '%s, #%d' % (data_name_pxi, self.ShotNo)
        plt.title(title, fontsize=18, loc='right')
        plt.show()
        '''

        filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        np.savetxt('radh_array' + filename, voltdata_array_org)
        np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

        '''
        _, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, 16, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403')
        _, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, 10, timedata[idx_time_st:idx_time_ed], -voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403')
        _, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, 5, timedata[idx_time_st:idx_time_ed], -voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403')

        #idex_offset = find_closest(timedata, 4.475)
        idex_offset = find_closest(timedata[idx_time_st:idx_time_ed], 4.5)

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dHz), #%d' % (fs, self.ShotNo)
        plt.title(title, fontsize=18, loc='right')
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr')
        #plt.pcolormesh(timedata[idx_time_st:idx_time_ed], r, (voltdata_array - np.mean(voltdata_array[idex_offset:idex_offset+10000,:], axis=0)).T, cmap='bwr')#, vmin=-0.1, vmax=0.1)
        #plt.pcolormesh(timedata[idx_time_st:idx_time_ed], r, voltdata_array.T, cmap='jet')
        plt.pcolormesh(timedata[idx_time_st:idx_time_ed], rho, voltdata_array.T, cmap='jet')
        #plt.pcolormesh(timedata[idx_time_st:idx_time_ed], r, (voltdata_array).T, cmap='bwr')#, vmin=-0.3, vmax=0.3)
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(timedata[idx_time_st:idx_time_ed-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(t, r, Te.T)
        #plt.pcolormesh(t, r[1:-1], (Te[:, 2:]-Te[:, 1:-1]-Te[:, 1:-1]+Te[:, :-2]).T, cmap='bwr')
        cb = plt.colorbar()
        #cb.set_label("d2Te/dr2")
        #cb.set_label(r"$V_{ECE}$ - offset")
        #cb.set_label("dTe/dt")
        cb.set_label("Te [0,1]")
        plt.xlabel("Time [sec]")
        #plt.xlim(4.4, 4.6)
        #plt.ylabel("R [m]")
        plt.ylabel("rho")
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        plt.show()
        '''



    def ana_plot_CTRLDISP1_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_name_ech = 'CTRLDISP1'
        timedata = TimeData.retrieve(data_name_ech, self.ShotNo, 1, 1).get_val()
        fig = plt.figure(figsize=(8,6), dpi=150)
        array_ECHRF = [13,14,16,17,22]
        name_gyrotron = ["#1", "#2", "#4", "#5", "#7"]
        i_name = 0
        #for i_ch in range(23):
        for i_ch in array_ECHRF:
            #data_ech = VoltData.retrieve(data_name_ech, self.ShotNo, 1, i_ch+1).get_val()
            data_ech = VoltData.retrieve(data_name_ech, self.ShotNo, 1, i_ch).get_val()
            #plt.plot(timedata, data_ech, label=name_gyrotron[i_name])
            plt.plot(timedata, data_ech - i_name*2, label=name_gyrotron[i_name])
            i_name += 1

        plt.vlines(5.0, -2, 6, colors='black')
        plt.xlim(4.8, 7.2)
        #plt.xlim(4.995, 5.01)
        #plt.xlim(5,7)
        #plt.ylim(-1,5.5)
        plt.legend()
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_ech, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlabel("Time [s]")
        plt.grid()
        finename = 'CTRLDISP1_%d.png' % self.ShotNo
        plt.savefig(finename)
        plt.close()
        plt.show()

    def ana_plot_PCI_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_fast = AnaData.retrieve('pci_ch4_fft_fast', self.ShotNo, 1)

        dnum = data_fast.getDimNo()
        for i in range(dnum):
            print(i, data_fast.getDimName(i), data_fast.getDimUnit(i))
        vnum = data_fast.getValNo()
        for i in range(vnum):
            print(i, data_fast.getValName(i), data_fast.getValUnit(i))
        timedata = data_fast.getDimData('Time')
        Sxx_50to150kHz = data_fast.getValData('50-150(kHz)')
        Sxx_150to500kHz = data_fast.getValData('150-500(kHz)')
        Sxx_50to500kHz = data_fast.getValData('50-500(kHz)')
        data_PCI_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_PCI_2D[:,i]= data_fast.getValData(i)





        interval_time_probe = 10
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)+7#+10#1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        #data_highK_condAV = data_highK[idx_time_st:idx_time_ed].reshape(np.int(mod_freq*(t_ed-t_st)), -1).T.mean(axis=-1)
        #data_DBS_condAV_ = data_DBS_2D[idx_time_st:idx_time_ed,:].reshape(idx_time_period, i_wave,vnum)
        #data_DBS_condAV = data_DBS_2D[idx_time_st:idx_time_ed,:].reshape(idx_time_period, i_wave, vnum).T.mean(axis=1)
        #data_DBS_condAV = data_DBS_2D[idx_time_st:idx_time_ed,:].reshape(idx_time_period, i_wave, vnum).mean(axis=1)
        #data_highK_condAV_mask = (data_highK[idx_time_st:idx_time_ed].reshape(i_wave, -1).T * data_highK_probe_condAV_01.T).mean(axis=-1)*i_wave/np.sum(data_highK_probe_condAV_01)
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(6,4), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_PCI_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i] - 500*i, label=data_fast.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = ' #%d, %.2fs-%.2fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='upper right')
        filename = "PCI_condAV_t%dto%dms_No%d.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

    def ana_plot_HIBP_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_name_HIBP = 'hibp_t_reff'
        data_HIBP = AnaData.retrieve(data_name_HIBP, self.ShotNo, 1)
        timedata = data_HIBP.getDimData("Time")
        vnum = data_HIBP.getValNo()
        data_HIBP_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_HIBP_2D[:,i]= data_HIBP.getValData(i)

        dnum = data_HIBP.getDimNo()
        for i in range(dnum):
            print(i, data_HIBP.getDimName(i), data_HIBP.getDimUnit(i))
        vnum = data_HIBP.getValNo()
        for i in range(vnum):
            print(i, data_HIBP.getValName(i), data_HIBP.getValUnit(i))


        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st# + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_HIBP_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(6,4), dpi=150)
        plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9)
        for i in range(vnum):
            data_HIBP_condAV[:,i] = data_HIBP_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            #plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i] - 5000*i, label=data_DBS.getValName(i))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        c = ax1.plot(timedata[:idx_time_period] - timedata[0], data_HIBP_condAV[:,3]*10, 'r-', label='Sum_signal*10', zorder=2)
        #c = ax1.plot(time, reffa99, '-', label='reff/a99', zorder=2)
        c = ax1.plot(timedata[:idx_time_period] - timedata[0], data_HIBP_condAV[:,4], 'g-', label='phi_pot', zorder=2)
        a = ax2.plot(timedata[:idx_time_period] - timedata[0], data_HIBP_condAV[:,6], 'b-', label='ntd', zorder=2)
        plt.title('HIBP, #%d, %.3f-%.3fs, reff/a99=%.4f' % (self.ShotNo,t_st, t_ed, np.mean(data_HIBP_condAV[:,2])), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax1.set_xlabel('Time [sec]')
        ax2.set_ylabel('ntd[a.u.]', rotation=270, labelpad=12)
        ax1.set_ylabel('phi_pot[kV(No calib.)]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        ax1.set_xlim(5,7)
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        filename = data_name_HIBP + "_condAV_t%dto%dms_No%d.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

    def ana_plot_DBS_Vp_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_name_DBS = 'mwrm_9o_R_Vp'
        #data_name_DBS = 'mwrm_comb_R_power'
        #data_name_DBS = 'mwrm_9o_R_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))


        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(10,15), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            #plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i] - 5000*i, label=data_DBS.getValName(i))
            if i % 1 == 0:
                plt.plot(timedata[:idx_time_period] - timedata[0], data_DBS_condAV[:, i], label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        plt.ylim(-2,2)
        #plt.xlim(0,0.1)
        plt.legend(loc='lower right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()


        '''
        data_name_DBS = 'mwrm_comb_R_power'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st# + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(10,15), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i] - 50*i, label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='lower right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

        data_name_DBS = 'mwrm_9o_R_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(6,4), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i], label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='lower right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()
        '''

        data_name_DBS = 'mwrm_comb_R_Vp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        for j in range(7):
            fig = plt.figure(figsize=(10,15), dpi=150)
            for i in range(7):
                i+=7*j
            #for i in range(vnum):
                data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
                #plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i], label=data_DBS.getValName(i))
                if i%1==0:
                    plt.plot(timedata[:idx_time_period] - timedata[0], data_DBS_condAV[:, i], label=data_DBS.getValName(i))
            #plt.xlim(4.4, 4.6)
            plt.xlabel('Time [sec]')
            title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
            plt.title(title, fontsize=18, loc='right')
            plt.xlim(0, timedata[idx_time_period]-timedata[0])
            plt.ylim(-2,2)
            #plt.xlim(0,0.1)
            plt.legend(loc='upper right')
            filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq_ChFreq%d_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo, j)
            plt.savefig(filename)
            plt.show()


        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])
        #return timedata[:idx_time_period]-timedata[0], reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1), y2
    def ana_plot_DBS_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_name_DBS = 'mwrm_9o_comb_Iamp'
        #data_name_DBS = 'mwrm_comb_R_power'
        #data_name_DBS = 'mwrm_9o_R_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))


        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(10,15), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            #plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i] - 5000*i, label=data_DBS.getValName(i))
            if i % 4 == 0:
                plt.plot(timedata[:idx_time_period] - timedata[0], data_DBS_condAV[:, i], label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='lower right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()


        '''
        data_name_DBS = 'mwrm_comb_R_power'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st# + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(10,15), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i] - 50*i, label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='lower right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

        data_name_DBS = 'mwrm_9o_R_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(6,4), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i], label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='lower right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()
        '''

        data_name_DBS = 'mwrm_comb_R_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(10,15), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            #plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i], label=data_DBS.getValName(i))
            if i%4==0:
                plt.plot(timedata[:idx_time_period] - timedata[0], data_DBS_condAV[:, i], label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='upper right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

        data_name_DBS = 'mwrm_comb_U_Iamp'
        data_DBS = AnaData.retrieve(data_name_DBS, self.ShotNo, 1)
        timedata = data_DBS.getDimData("Time")
        vnum = data_DBS.getValNo()
        data_DBS_2D = np.zeros((len(timedata), vnum))
        for i in range(vnum):
            data_DBS_2D[:,i]= data_DBS.getValData(i)

        dnum = data_DBS.getDimNo()
        for i in range(dnum):
            print(i, data_DBS.getDimName(i), data_DBS.getDimUnit(i))
        vnum = data_DBS.getValNo()
        for i in range(vnum):
            print(i, data_DBS.getValName(i), data_DBS.getValUnit(i))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_DBS_condAV = np.zeros((idx_time_period, vnum))

        fig = plt.figure(figsize=(10,15), dpi=150)
        for i in range(vnum):
            data_DBS_condAV[:,i] = data_DBS_2D[idx_time_st:idx_time_ed,i].reshape(i_wave, -1).T.mean(axis=-1)
            #plt.plot(timedata[:idx_time_period]-timedata[0], data_DBS_condAV[:,i], label=data_DBS.getValName(i))
            if i%4==0:
                plt.plot(timedata[:idx_time_period] - timedata[0], data_DBS_condAV[:, i], label=data_DBS.getValName(i))
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_DBS, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend(loc='upper right')
        filename = data_name_DBS + "_condAV_t%dto%dms_No%d_lowfreq.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])
        #return timedata[:idx_time_period]-timedata[0], reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1), y2

    def ana_plot_MPbandpass_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        data_mp_2d = data_mp.getValData(0)
        timedata = data_mp.getDimData('TIME')
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))


        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        i_ch =0
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        data_mp = data_mp_2d[idx_time_st:idx_time_ed, i_ch]

        # サンプリング周波数の計算
        fs = 1 / (timedata[1] - timedata[0])
        # フーリエ変換
        fft_result = fft(data_mp)
        freqs = fftfreq(len(data_mp), 1 / fs)

        # 周波数スペクトルのプロット
        plt.figure(figsize=(6, 6))
        plt.plot(freqs[:len(freqs) // 2], np.abs(fft_result)[:len(freqs) // 2])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.yscale('log')
        title = 'MP #%d, t=%.4f-%.4fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        filename = 'Freq_Spectrum_MP_SN%d.png' % (self.ShotNo)
        plt.savefig(filename)
        plt.show()

        # スペクトログラムの計算
        #nperseg =  2**8# 窓長さ
        #noverlap =  2**3  # 窓のオーバーラップ
        #nperseg = 2000  # 窓長さ  for 10Hz MECH
        #noverlap = 1950  # 窓のオーバーラップ
        #noverlap = 1990  # 窓のオーバーラップ
        nperseg = 100  # 窓長さ    for 40Hz MECH
        noverlap = 90  # 窓のオーバーラップ
        frequencies, times, Sxx = sig.spectrogram(data_mp, fs, nperseg=nperseg, noverlap=noverlap)

        idx_time_period_Sxx = int(len(times-1)/i_wave)

        #Sxx_condAV = Sxx[:,:-1].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        Sxx_condAV = Sxx[:,:-11].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)

        # スペクトログラムのプロット
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times + t_st, frequencies, 10 * np.log10(Sxx), cmap='jet')#, vmin=-100, vmax=-70)#shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        title = 'MP #%d, t=%.4f-%.4fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim([100, 50000])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([0, 20000])  # 表示範囲を0から10kHzまでに設定
        plt.xlim(5.2,5.8)
        plt.yscale('log')
        filename = 'Spectrogram_MP_SN%d.png' % (self.ShotNo)
        plt.savefig(filename)
        plt.show()

        # スペクトログラムのプロット
        plt.figure(figsize=(10, 12))
        plt.pcolormesh(times[:idx_time_period_Sxx]-0.01, frequencies, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-90, vmax=-30)#shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        title = 'Cond. AV. MP #%d, t=%.4f-%.4fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([100, 50000])  # 表示範囲を0から10kHzまでに設定
        plt.yscale('log')
        #plt.xlim(5.1,5.5)
        filename = 'Spectgram_condAV_MP_SN%d_v2.png' % (self.ShotNo)
        plt.savefig(filename)
        plt.show()

        # バターワースバンドパスフィルタの設計
        lowcut_1 = 100  # カットオフ周波数 1kHz
        highcut_1 = 2700  # カットオフ周波数 10kHz
        label_lowfreq = '%.1f-%.1f kHz' % (lowcut_1/1000, highcut_1/1000)
        order = 3#5
        nyquist = 0.5 * fs
        low = lowcut_1 / nyquist
        high = highcut_1 / nyquist
        b, a = butter(order, [low, high], btype='band')

        # 複素振幅強度にフィルタを適用
        filtered_mp = filtfilt(b, a, data_mp)
        data_mp_env = np.abs(signal.hilbert(data_mp))
        filtered_mp_env = np.abs(signal.hilbert(filtered_mp))

        data_mp_env_condAV = data_mp_env.reshape(i_wave, -1).T.mean(axis=-1)
        filtered_mp_env_condAV_lowfreq = filtered_mp_env.reshape(i_wave, -1).T.mean(axis=-1)

        # バターワースバンドパスフィルタの設計
        #lowcut = 10000#1000  # カットオフ周波数 1kHz
        #highcut = 45000#10000  # カットオフ周波数 10kHz
        lowcut_2 = 20000#1000  # カットオフ周波数 1kHz
        highcut_2 = 45000#10000  # カットオフ周波数 10kHz
        label_highfreq = '%.1f-%.1f kHz' % (lowcut_2/1000, highcut_2/1000)
        order = 3#5
        # サンプリング周波数の計算
        fs = 1 / (timedata[1] - timedata[0])
        nyquist = 0.5 * fs
        low = lowcut_2 / nyquist
        high = highcut_2 / nyquist
        b, a = butter(order, [low, high], btype='band')

        # 複素振幅強度にフィルタを適用
        filtered_mp = filtfilt(b, a, data_mp)
        filtered_mp_env = np.abs(signal.hilbert(filtered_mp))

        filtered_mp_env_condAV_highfreq = filtered_mp_env.reshape(i_wave, -1).T.mean(axis=-1)

        fig = plt.figure(figsize=(6,4), dpi=150)
        #plt.plot(timedata[idx_time_st:idx_time_ed], data_mp, label='raw')
        #plt.plot(timedata[idx_time_st:idx_time_ed], filtered_mp, label='filtered')
        #plt.plot(timedata[idx_time_st:idx_time_ed], filtered_mp_env, label='filtered(envelope)')
        plt.plot(timedata[:idx_time_period-1]-timedata[0], data_mp_env_condAV, label='raw(envelope)')
        plt.plot(timedata[:idx_time_period-1]-timedata[0], filtered_mp_env_condAV_lowfreq, label='filtered(Low freq.:' + label_lowfreq + ', envelope)')
        plt.plot(timedata[:idx_time_period-1]-timedata[0], filtered_mp_env_condAV_highfreq, label='filtered(High freq.:' + label_highfreq + ', envelope)')
        plt.legend()
        title = '#%d, t=%.4f-%.4fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [sec]')
        plt.xlim(0, 1/mod_freq)
        #plt.xlim(5.1, 5.5)
        filename = 'CondAv_MP_SN%d_t%dto%dms_BP%dto%d_%dto%dkHz.png' % (self.ShotNo, t_st*1000, t_ed*1000, lowcut_1/1000, highcut_1/1000, lowcut_2/1000, highcut_2/1000)
        plt.savefig(filename)
        plt.show()

    def ana_plot_gradTe_vs_highK_MECH(self, t_st, t_ed, mod_freq):
        _, rho_highK, _ = self.ana_plot_highK_condAV_MECH(t_st+0.1, t_ed, mod_freq)
        timedata_gradTe, rho_gradTe, data_gradTe = self.ana_plot_radhpxi_calThom_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK, data_highK = self.ana_plot_BSADC_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK *= 1e3
        timedata_highK_ = np.linspace(0, 1/mod_freq, len(timedata_highK))
        timedata_gradTe_ = np.linspace(0, 1/mod_freq, len(timedata_gradTe))
        f = interp1d(timedata_highK_, data_highK, kind='cubic')
        data_highK_extended = f(timedata_gradTe_)

        i_rho = np.abs(rho_gradTe - rho_highK).argmin()
        print(i_rho)
        #i_rho = 20

        x = -data_gradTe[:,i_rho]  # 説明変数
        y = data_highK_extended  # 目的変数
        # -----------------------------------------------------------
        # 1. 時刻 0.035–0.060 s を除外
        # -----------------------------------------------------------
        mask = (timedata_gradTe_ < 0.0027) | (timedata_gradTe_ > 0.0056)  # True なら採用
        t_f, x_f, y_f = timedata_gradTe_[mask], x[mask], y[mask]
        plt.plot(timedata_gradTe_,y)
        plt.plot(t_f,y_f)
        plt.show()

        if t_f.size < 3:  # 念のため
            raise ValueError("使えるデータ点が 3 未満です。")
        r,p = pearsonr(x_f, y_f)
        print(f"Pearson r = {r:.6f},  p-value = {p:.6e}")

        slope, intercept, r_value, p_value, stderr = linregress(x_f, y_f)
        R2 = r_value ** 2
        print(f"y ≈ {intercept:.6g} + {slope:.6g}·x  (R² = {R2:.6f})")

        k = np.sum(x_f * y_f) / np.sum(x_f ** 2)  # 最小二乗で傾き k (切片 0)
        residuals = y_f - k * x_f
        R2_prop = 1 - residuals.var() / y_f.var()
        print(f"比例係数 k = {k:.6g},  R² (through 0) = {R2_prop:.36}")

        label_gradTe = 'gradTe(r/a=%.3f)' % rho_gradTe[i_rho]
        label_highK = 'highK(r/a=%.3f)' % rho_highK

        # --- プロット --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        # カラーマップ設定（ここでは "viridis"）
        norm = plt.Normalize(vmin=t_f.min(), vmax=t_f.max())
        sc = ax.scatter(x_f, y_f, c=t_f, cmap='viridis', norm=norm,
                        s=10, alpha=0.8, label='data')

        # 回帰線
        xp = np.linspace(x_f.min(), x_f.max(), 200)
        ax.plot(xp, slope * xp + intercept, 'r-', lw=1.2, label='linear fit')
        ax.plot(xp, k * xp, 'g--', lw=1.0, label='prop fit')

        ax.set_xlabel('data_gradTe')
        ax.set_ylabel('data_highK')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')

        # --- ④ カラーバー（時間） --------------------------------------------
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'),
                            ax=ax, pad=0.02, aspect=35)
        cbar.set_label('Time [s]')

        # --- 枠外に統計テキスト ---------------------------------------------
        info = (
        f"#{self.ShotNo}, {t_st:.2f}s-{t_ed:.2f}s\n"
        f"r/a={rho_gradTe[i_rho]:.3g}(gradTe), {rho_highK:.3g}(highK)\n"
        f"Pearson  r  = {r:.3f}\n"
            f"p-value      < {np.finfo(float).tiny:.0e}\n"  # p=0 対策
            f"Linear  y = {intercept:.3g} + {slope:.3g}·x\n"
            f"R² (linear) = {R2:.3f}\n"
            f"Proportional k = {k:.3g}\n"
            f"R² (y=kx)   = {R2_prop:.3f}"
        )
        # 左上余白に配置（0–1 は Figure 座標）
        fig.text(0.15, 0.98, info, va='top', ha='left',
                 fontsize=9, family='monospace')

        # レイアウト調整：右・上に 10% ずつ余白
        plt.tight_layout(rect=[0, 0, 0.95, 0.8])
        filename = 'highK_vs_gradTe_SN%d_lowfreq.png' % (self.ShotNo)
        plt.savefig(filename)
        plt.show()

        fig, ax1 = plt.subplots(figsize=(6, 4))

        # ---- 左軸：gradTe ----
        ln1, = ax1.plot(timedata_gradTe_, -data_gradTe[:, i_rho],
                        color='tab:blue', lw=1.5, label='gradTe(' + label_gradTe)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('gradTe', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # ---- 右軸：highK ----
        ax2 = ax1.twinx()  # 右軸を追加
        ln2, = ax2.plot(timedata_gradTe_, data_highK_extended,
                        color='tab:red', lw=1.5, label='highK(' + label_highK)
        ax2.set_ylabel('highK', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # ---- 仕上げ ----
        # 横軸範囲を両データの共通領域に合わせたい場合
        #xmin = min(timedata_gradTe_.min(), timedata_highK_.min())
        #xmax = max(timedata_gradTe_.max(), timedata_highK_.max())
        #ax1.set_xlim(xmin, xmax)

        # ２系列の凡例をまとめて表示
        ax1.legend(handles=[ln1, ln2], loc='upper right')

        ax1.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        

        plt.plot(-data_gradTe[:, i_rho],data_highK_extended , label='gradTe('+label_gradTe + ',highK(' + label_highK)
        plt.xlim(0,)
        plt.ylim(0,)
        plt.legend()
        plt.show()

    def cal_phase_diff_BS_ECE_MECH_BS1ch(self, t_st, t_ed, mod_freq):
        _, rho_highK, _ = self.ana_plot_highK_condAV_MECH(t_st+0.1, t_ed, mod_freq)
        timedata_gradTe, rho_gradTe, data_gradTe = self.ana_plot_radhpxi_calThom_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK, data_highK = self.ana_plot_BSADC_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK *= 1e3
        timedata_highK_ = np.linspace(0, 1/mod_freq, len(timedata_highK))
        timedata_gradTe_ = np.linspace(0, 1/mod_freq, len(timedata_gradTe))
        f = interp1d(timedata_highK_, data_highK, kind='cubic')
        data_highK_extended = f(timedata_gradTe_)
        Te_all = data_gradTe
        n_all = data_highK_extended

        # -----------------------
        # 0. 解析パラメータ
        # -----------------------
        #FS = 100_000  # サンプリング周波数 [Hz]
        FS = 1/(timedata_gradTe_[1] - timedata_gradTe_[0])
        BAND_HP = (5, 15)  # ヒートパルス成分抽出帯域 [Hz]
        BAND_LF = (10_000, 20_000)  # 低-f 密度揺動抽出帯域 [Hz]
        LP_ENVELOPE = 50  # 包絡線変調を残すローパスカットオフ [Hz]
        NTAP = 2048  # FIR フィルタタップ数（奇数）
        TRIG_FREQ = 10  # 加熱周波数 [Hz]
        WIN_CYC = int(FS / TRIG_FREQ)  # 1 サイクル点数
        BAD_START_MS = 2.7  # [ms]  ← ここから
        BAD_END_MS = 5.6  # [ms]  ← ここまで除外
        idx_bad0 = int(BAD_START_MS * 1e-3 * FS)  # サンプル番号に変換
        idx_bad1 = int(BAD_END_MS * 1e-3 * FS)
        # -----------------------
        Nt = len(data_highK_extended)
        trig_all = np.arange(0, Nt, WIN_CYC)  # きっちり 10 Hz の場合
        R_te = rho_gradTe
        R_n = rho_highK
        r_n = rho_highK

        # Te 側で最も近いチャンネルを自動選択
        idx_te = int(np.argmin(np.abs(R_te - r_n)))
        print(f'low-f (R={r_n:.3f} m)  <->  Te ch {idx_te} (R={R_te[idx_te]:.3f} m)')

        # 1 次元 → 2 D 形状 (Nt,1) にして共通関数へ
        te1 = Te_all[:, idx_te][:, None]
        n1 = data_highK_extended
        # ====== 2. 位相抽出 ======
        phi_te = phase_te(te1)  # (Nt,1)
        phi_lf = phase_lowf(n1)  # (Nt,1)

        # ====== 3. トリガ基準で切り出し ======
        te_cycle = cut_cycles(phi_te, trig_all)  # (Ncyc, win, 1)
        lf_cycle = cut_cycles(phi_lf, trig_all)  # (Ncyc, win, 1)

        # --- cut_cycles の直後 ---
        te_cycle = cut_cycles(phi_te, trig_all)  # (Ncyc, win, N_te)
        lf_cycle = cut_cycles(phi_lf, trig_all)  # (Ncyc, win, N_lf)  ← ここまで従来通り

        # 2.7–5.6 ms 区間を無効化
        lf_cycle[:, idx_bad0:idx_bad1, :] = np.nan

        # ====== 4. 位相差・ラグ ======
        dphi, dlag = phase_diff_lag(te_cycle, lf_cycle)  # 2D (Ncyc, win)

        dphi_mean = np.nanmean(dphi, axis=0)  # (-, win)   ← NaN 無視
        dlag_mean = np.nanmean(dlag, axis=0)
        # ====== 5. 平均と可視化 ======
        dphi_mean = np.nanmean(dphi, axis=0)  # (win,)
        dlag_mean = np.nanmean(dlag, axis=0)

        #t_axis = np.linspace(0, 1 / TRIG_FREQ, WIN_CYC, endpoint=False) * 1e3  # [ms]
        # 位相差・ラグを算出したあと
        dphi_mean = np.nanmean(dphi, axis=0)  # (win_actual,)
        dlag_mean = np.nanmean(dlag, axis=0)

        # サンプル長に合わせて時間軸を生成
        win_actual = dphi_mean.shape[0]  # 切り出せた点数
        t_axis = np.arange(win_actual) / FS * 1e3  # [ms]

        plt.figure(figsize=(5, 3))
        plt.plot(t_axis, dphi_mean)
        plt.ylabel('<Δφ> [rad]')
        plt.xlabel('Time [ms]')
        plt.title('Phase diff (Te - low-f)')

        plt.figure(figsize=(5, 3))
        plt.plot(t_axis, dlag_mean)
        plt.ylabel('<Δt> [ms]')
        plt.xlabel('Time [ms]')
        plt.title('Lag (negative → low-f advance)')
        plt.tight_layout()
        plt.show()

    def cal_phase_diff_BS_ECE_MECH(self, t_st, t_ed, mod_freq):
        _, rho_highK, _ = self.ana_plot_highK_condAV_MECH(t_st+0.1, t_ed, mod_freq)
        timedata_gradTe, rho_gradTe, data_gradTe = self.ana_plot_radhpxi_calThom_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK, data_highK = self.ana_plot_BSADC_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK *= 1e3
        timedata_highK_ = np.linspace(0, 1/mod_freq, len(timedata_highK))
        timedata_gradTe_ = np.linspace(0, 1/mod_freq, len(timedata_gradTe))
        f = interp1d(timedata_highK_, data_highK, kind='cubic')
        data_highK_extended = f(timedata_gradTe_)
        Te_all = data_gradTe
        n_all = data_highK_extended

        # -----------------------
        # 0. 解析パラメータ
        # -----------------------
        #FS = 100_000  # サンプリング周波数 [Hz]
        FS = 1/(timedata_gradTe_[1] - timedata_gradTe_[0])
        BAND_HP = (5, 15)  # ヒートパルス成分抽出帯域 [Hz]
        BAND_LF = (10_000, 20_000)  # 低-f 密度揺動抽出帯域 [Hz]
        LP_ENVELOPE = 50  # 包絡線変調を残すローパスカットオフ [Hz]
        NTAP = 2048  # FIR フィルタタップ数（奇数）
        TRIG_FREQ = 10  # 加熱周波数 [Hz]
        WIN_CYC = int(FS / TRIG_FREQ)  # 1 サイクル点数
        # -----------------------
        Nt = len(data_highK_extended)
        trig_all = np.arange(0, Nt, WIN_CYC)  # きっちり 10 Hz の場合
        R_te = rho_gradTe
        R_n = rho_highK
        # 2. 位相抽出
        phi_te = phase_te(Te_all)
        phi_lf = phase_lowf(n_all)

        # 3. サイクル切り出し
        te_cycle = cut_cycles(phi_te, trig_all)
        lf_cycle = cut_cycles(phi_lf, trig_all)

        # 4. 位相差・遅れ
        dphi, dlag = phase_diff_lag(te_cycle, lf_cycle)

        # 5. Cycle 平均
        dphi_mean = dphi.mean(axis=0)  # (win, N_R_common)
        dlag_mean = dlag.mean(axis=0)

        # 6. コヒーレンス (10 Hz)
        sig_te_10hz = _fir_bandpass(Te_all[:, 0], *BAND_HP, FS)  # ECE 1 ch
        sig_nf_10hz = _fir_bandpass(n_all[:, 0], *BAND_HP, FS)  # 低-f 1 ch

        f_coh, coh = coherence(sig_te_10hz, sig_nf_10hz,
                               FS, nperseg=WIN_CYC)

        idx10 = np.argmin(np.abs(f_coh - TRIG_FREQ))
        gamma2 = coh[idx10]  # コヒーレンス γ²(10 Hz)
        print(f'Coherence γ²(10 Hz): {gamma2:.2f}')

        # -----------------------
        # 7. 可視化
        # -----------------------
        t_axis = np.linspace(-0.5 / TRIG_FREQ, 0.5 / TRIG_FREQ, WIN_CYC) * 1e3  # [ms]

        plt.figure(figsize=(6, 4))
        plt.pcolormesh(R_te, t_axis, dphi_mean, shading='auto', cmap='twilight',
                       vmin=-np.pi, vmax=np.pi)
        plt.colorbar(label='Δφ [rad]')
        plt.xlabel('R [m]')
        plt.ylabel('Time [ms]')
        plt.title('Phase difference map (Te - low-f)')
        plt.tight_layout()

        plt.figure(figsize=(6, 4))
        plt.pcolormesh(R_te, t_axis, dlag_mean, shading='auto', cmap='bwr',
                       vmin=-10, vmax=10)
        plt.colorbar(label='Δt [ms]')
        plt.xlabel('R [m]')
        plt.ylabel('Time [ms]')
        plt.title('Lag map (Te behind low-f → negative)')
        plt.tight_layout()

        print(f'Coherence γ²(10 Hz) example channel: {gamma2:.2f}')
        plt.show()

    def ana_plot_BSADC_condAV_MECH_org(self, t_st, t_ed, mod_freq, isSaveData=False, isFitting=False):
        #ch1(4.0m)
        #ch_I = 17
        #ch_Q = 18
        #ch2(4.4m)
        #ch_I = 23G
        #ch_Q = 24
        #ch3
        ch_I = 19
        ch_Q = 20
        #ch4(140G,4.4m)
        #ch_I = 21
        #ch_Q = 22
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        #ch_I = 9
        #data_name_BS = 'MWRM-PXI'
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_BS, start_time)
        end_index = np.searchsorted(timedata_BS, end_time)
        filtered_time = timedata_BS[start_index:end_index]
        filtered_signal = voltdata_BS_I[start_index:end_index]

        # フーリエ変換
        fft_result = fft(filtered_signal)
        freqs = fftfreq(len(filtered_signal), 1 / fs)

        # 周波数スペクトルのプロット
        plt.figure(figsize=(6, 6))
        plt.plot(freqs[:len(freqs) // 2], np.abs(fft_result)[:len(freqs) // 2])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.yscale('log')
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        filename = 'Freq_Spectrum_BS_%s_ch%d_SN%d.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.savefig(filename)
        plt.show()

        # スペクトログラムの計算
        #nperseg =  12500# 窓長さ
        #noverlap =  512  # 窓のオーバーラップ
        #nperseg =  2**15# 窓長さ
        #noverlap =  2**8  # 窓のオーバーラップ
        #nperseg =  50000# 窓長さ
        #noverlap =  47500  # 窓のオーバーラップ
        #nperseg =  50000# 窓長さ   for 10Hz MECH
        #noverlap =  48750  # 窓のオーバーラップ
        #noverlap =  49750  # 窓のオーバーラップ
        nperseg =  5000# 窓長さ   for 10Hz MECH
        noverlap =  4750  # 窓のオーバーラップ
        #nperseg = 2500  # 窓長さ   for 40Hz MECH
        #noverlap = 2250  # 窓のオーバーラップ
        frequencies, times, Sxx = sig.spectrogram(filtered_signal, fs, nperseg=nperseg, noverlap=noverlap)

        # スペクトログラムのプロット
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times + start_time, frequencies, 10 * np.log10(Sxx), cmap='jet')#, vmin=-100, vmax=-70)#shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        #plt.ylim([0, 80000])  # 表示範囲を0から10kHzまでに設定
        plt.xlim(5.1,5.5)
        plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.yscale('log')
        filename = 'Spectgram_BS_%s_ch%d_SN%d.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.savefig(filename)
        plt.show()

        '''
        # スケールの範囲を設定（必要な周波数帯域に応じて調整）
        scales = np.arange(1, 5000)  # スケール範囲を広げて低周波数成分もカバー

        # チャンクサイズとオーバーラップを設定
        chunk_size = 100000  # 例: 10万サンプルのチャンク
        overlap = 50000  # 例: 5万サンプルのオーバーラップ

        # 結果を保存するリストを初期化
        coef_list = []
        time_list = []

        # チャンク数を計算
        total_chunks = (len(filtered_signal) - overlap) // (chunk_size - overlap) + 1

        # 処理開始時間
        total_start_time = time.time()

        # チャンクごとに処理
        for i, start in enumerate(range(0, len(filtered_signal), chunk_size - overlap)):
            end = min(start + chunk_size, len(filtered_signal))
            chunk_signal = filtered_signal[start:end]
            chunk_time = filtered_time[start:end]

            # チャンク処理開始時間
            start_process_time = time.time()

            # 連続ウェーブレット変換 (CWT)
            coef, freqs = pywt.cwt(chunk_signal, scales, 'morl', sampling_period=1 / fs)

            # 結果をリストに追加
            coef_list.append(coef)
            time_list.append(chunk_time)

            # チャンク処理終了時間
            end_process_time = time.time()

            # 進捗状況と処理時間をコンソールに表示
            process_time = end_process_time - start_process_time
            elapsed_time = end_process_time - total_start_time
            remaining_chunks = total_chunks - (i + 1)
            estimated_total_time = elapsed_time / (i + 1) * total_chunks
            estimated_remaining_time = estimated_total_time - elapsed_time
            progress_percent = ((i + 1) / total_chunks) * 100

            print(f"Chunk {i + 1}/{total_chunks} processed in {process_time:.2f} seconds")
            print(
                f"Elapsed time: {elapsed_time:.2f} seconds, Estimated remaining time: {estimated_remaining_time:.2f} seconds, Progress: {progress_percent:.2f}%")

        # 結果を結合
        coef_combined = np.hstack(coef_list)
        time_combined = np.hstack(time_list)

        # スペクトログラムのプロット
        plt.figure(figsize=(12, 6))
        plt.contourf(time_combined, freqs, np.abs(coef_combined), extend='both', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform (5.1s-5.6s)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim([0, 10000])  # 表示範囲を0から10kHzまでに設定
        plt.show()
        '''

        # フーリエ変換
        fft_result = fft(filtered_signal)

        # 複素振幅の計算
        complex_amplitude = ifft(fft_result)

        # 複素振幅の絶対値を計算（複素振幅強度）
        amplitude_strength = np.abs(complex_amplitude)

        # バターワースバンドパスフィルタの設計
        #lowcut = 90000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        #lowcut = 5000#1000  # カットオフ周波数 1kHz
        #highcut = 30000#10000  # カットオフ周波数 10kHz
        #lowcut = 50000#50000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        lowcut = 10000  # カットオフ周波数 1kHz
        highcut = 20000  # カットオフ周波数 10kHz
        #lowcut = 500  # カットオフ周波数 1kHz
        #lowcut = 1000  # カットオフ周波数 1kHz
        #highcut = 2700  # カットオフ周波数 10kHz
        #highcut = 10000  # カットオフ周波数 10kHz
        order = 2#3#5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        #b, a = butter(order, low, btype='low', analog=False)

        # 複素振幅強度にフィルタを適用
        filtered_amplitude_strength = np.abs(filtfilt(b, a, amplitude_strength))



        idx_time_st = find_closest(timedata_BS, t_st)
        idx_time_ed = find_closest(timedata_BS, t_ed)+1
        idx_time_period = find_closest(timedata_BS, t_st + 1/mod_freq) - idx_time_st + 1
        #i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        i_wave = int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        idx_time_period_Sxx = int(len(times-21)/i_wave)

        Sxx_condAV = Sxx[:,:-1].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        #Sxx_condAV = Sxx[:,:-11].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        #Sxx_condAV = Sxx[:,:-21].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        # スペクトログラムのプロット
        #plt.figure(figsize=(4, 6*8.4/9), dpi=150)
        plt.figure(figsize=(4.1, 6), dpi=150)
        #plt.pcolormesh(times[:idx_time_period_Sxx]-0.01, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-120, vmax=-105)#shading='gouraud')
        plt.pcolormesh(times[:idx_time_period_Sxx]*1000, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-120, vmax=-105)#shading='gouraud')
        cbar = plt.colorbar(pad=0.02)  # <- カラーバーオブジェクトを変数に格納
        cbar.set_label("Intensity [dB]", fontsize=20)  # <- ラベル設定
        cbar.ax.tick_params(labelsize=16)  # <- 目盛りフォントサイズ設定
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [ms]', fontsize=20)
        plt.ylabel('Frequency [kHz]', fontsize=20)
        #plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.ylim([0, 100])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([30, 500000])  # 表示範囲を0から10kHzまでに設定
        #plt.yscale('log')
        plt.xlim(0,)
        plt.tick_params(labelsize=16)
        plt.xticks([0, 20, 40, 60, 80, 100])
        filename = 'Spectgram_condAV_BS_%s_ch%d_SN%d_v4.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

        #data_BS_condAV = filtered_amplitude_strength[idx_time_st:idx_time_ed].reshape(i_wave, -1).T.mean(axis=-1)
        data_BS_condAV = filtered_amplitude_strength.reshape(i_wave, -1).T.mean(axis=-1)
        #data_BS_condAV = amplitude_strength.reshape(i_wave, -1).T.mean(axis=-1)
        fs = 2e3#2e2#2e3#5e3#5e3#1e3
        dt = timedata_BS[1]-timedata_BS[0]
        freq = fftpack.fftfreq(len(timedata_BS[:idx_time_period-1]), dt)
        yf = fftpack.rfft(data_BS_condAV) / (int(len(data_BS_condAV) / 2))
        yf2 = np.copy(yf)
        yf2[(freq > fs)] = 0
        yf2[(freq < 0)] = 0
        y2 = np.real(fftpack.irfft(yf2) * (int(len(data_BS_condAV) / 2)))

        # 最大値とその時刻を取得
        max_value = np.max(y2)
        max_index = np.argmax(y2)
        time = timedata_BS[:idx_time_period-1]-timedata_BS[0]
        max_time = time[max_index]

        print(f"Max value: {max_value} \nMax time: {max_time}")

        if isFitting:
            # 時刻0から0.025sまでのデータを選択
            start_time = 0.035#0
            end_time = 0.06#0.1#0.007#0.025
            timedata_condAV = timedata_BS[:idx_time_period-1]-timedata_BS[0]
            mask = (timedata_condAV >= start_time) & (timedata_condAV <= end_time)
            filtered_timedata = timedata_condAV[mask]
            filtered_data = y2[mask]

            # 0.003sから0.005sのデータを除外
            exclude_start_time = 0.0029
            exclude_end_time = 0.0051
            exclude_mask = (filtered_timedata >= exclude_start_time) & (filtered_timedata <= exclude_end_time)
            final_timedata = filtered_timedata[~exclude_mask]
            final_data = filtered_data[~exclude_mask]

            # フィッティング
            initial_guess = [0, 0.001, 0.04, 0.0005]  # 初期パラメータの推定値
            #initial_guess = [0, 0.001, 0.020, 0.0005]  # 初期パラメータの推定値
            params, covariance = curve_fit(extreme_func, final_timedata, final_data, p0=initial_guess)
            perr = np.sqrt(np.diag(covariance))
            #perr =np.diag(params)

            # フィッティング結果のパラメータ
            y0_fit, A_fit, xc_fit, w_fit = params
            y0_fit_err, A_fit_err, xc_fit_err, w_fit_err = perr
            print(f"Fitted parameters for Extreme: \ny0 = {y0_fit}±{y0_fit_err}, \nA = {A_fit}±{A_fit_err}, \nxc = {xc_fit}±{xc_fit_err}, \nw = {w_fit}±{w_fit_err}")
            # フィッティング曲線を生成
            fit_curve = extreme_func(timedata_condAV, *params)

            # 既に同じショット番号が存在するかチェック
            existing_shots = set()
            file_path = "fitting_parameter_Extreme_50to100kHz_20240509.csv"
            header = ['ShotNo', 'reff_a99_highK', 'y0', 'y0_error', 'A','A_error', 'xc', 'xc_error', 'w',  'w_error']
            # 既存データの読み込み
            data_rows = []
            if os.path.exists(file_path):
                with open(file_path, mode='r', newline='') as file:
                    reader = csv.reader(file)
                    data_rows = list(reader)

            # ヘッダーが存在しない場合は追加
            if not data_rows:
                data_rows.append(header)

            data_name_highK = 'mwrm_highK_Power3'  # RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
            reff_a99_highK = data.getValData('reff/a99')
            reff_a99_highK_av = reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1)
            # 同じショット番号のデータが存在するかチェックし、存在する場合は上書き
            new_row = [self.ShotNo, reff_a99_highK_av, y0_fit, y0_fit_err, A_fit, A_fit_err, xc_fit,  xc_fit_err, w_fit, w_fit_err]
            shot_exists = False
            for i, row in enumerate(data_rows):
                if row[0] == str(self.ShotNo):
                    data_rows[i] = new_row
                    shot_exists = True
                    break

            if not shot_exists:
                data_rows.append(new_row)

            # ファイルに書き込む
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data_rows)

            print(f"Data for shot number {self.ShotNo} has been written to the file.")


        plt.plot(filtered_time, filtered_amplitude_strength)
        #plt.plot(filtered_time, amplitude_strength)
        #plt.plot(timedata_BS, voltdata_BS_Q)
        #plt.plot(phase_BS)
        #plt.xlim(5,5.0001)
        #plt.xlim(5,5.1)
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.show()

        fig = plt.figure(figsize=(6,6), dpi=150)
        if isFitting:
            plt.subplots_adjust(top=0.75)
        plt.plot(timedata_BS[:idx_time_period-1]-timedata_BS[0], data_BS_condAV)
        plt.plot(timedata_BS[:idx_time_period-1]-timedata_BS[0], y2)
        plt.plot(max_time, max_value, 'ro', label='Max point')
        title = 'Cond. Av., %s(ch.:%d), #%d, t=%.3f-%.3fs, \nBnad-pass: %.1f-%.1fkHz' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
        plt.xlabel('Time[sec]')
        if isFitting:
            plt.plot(timedata_condAV, fit_curve, 'r-', linewidth=2, label='Extreme fit')
            title_2 = f"\nFitted parameters for Extreme: \ny0 = {y0_fit}±{y0_fit_err}, \nA = {A_fit}±{A_fit_err}, \nxc = {xc_fit}±{xc_fit_err}, \nw = {w_fit}±{w_fit_err}"
            plt.title(title + title_2, fontsize=13, loc='right')
            plt.legend()
        else:
            plt.title(title)

        #plt.xlim(0,0.01)
        plt.xlim(0, 1/mod_freq)
        #plt.xlim(0, 0.1)
        #plt.ylim(0,0.0005)
        plt.ylim(0,)
        filename = 'CondAv_BS_%s_ch%d_SN%d_t%dto%dms_Bnadpass_%.1fto%.1fkHz_fitting.png' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
        plt.savefig(filename)
        plt.show()
        if isSaveData:
            t_data_BS = np.vstack((timedata_BS[:idx_time_period-1]-timedata_BS[0], y2))
            filename = 'CondAv_BS_%s_ch%d_SN%d_t%dto%dms_Bnadpass_%.1fto%.1fkHz.txt' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
            np.savetxt('time_' + filename, t_data_BS.T)

        return timedata_BS[:idx_time_period-1]-timedata_BS[0], y2

    def ana_plot_BSADC_condAV_MECH(self, t_st, t_ed, mod_freq, isSaveData=False, isFitting=False, isBSTwoBand=False):
        #ch1(4.0m)
        #ch_I = 17
        #ch_Q = 18
        #ch2(4.4m)
        #ch_I = 23G
        #ch_Q = 24
        #ch3
        ch_I = 19
        ch_Q = 20
        #ch4(140G,4.4m)
        #ch_I = 21
        #ch_Q = 22
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        #ch_I = 9
        #data_name_BS = 'MWRM-PXI'
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_BS, start_time)
        end_index = np.searchsorted(timedata_BS, end_time)
        filtered_time = timedata_BS[start_index:end_index]
        filtered_signal = voltdata_BS_I[start_index:end_index]

        # フーリエ変換
        fft_result = fft(filtered_signal)
        freqs = fftfreq(len(filtered_signal), 1 / fs)

        # 周波数スペクトルのプロット
        plt.figure(figsize=(6, 6))
        plt.plot(freqs[:len(freqs) // 2], np.abs(fft_result)[:len(freqs) // 2])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.yscale('log')
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        filename = 'Freq_Spectrum_BS_%s_ch%d_SN%d.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.savefig(filename)
        plt.show()

        # スペクトログラムの計算
        #nperseg =  12500# 窓長さ
        #noverlap =  512  # 窓のオーバーラップ
        #nperseg =  2**15# 窓長さ
        #noverlap =  2**8  # 窓のオーバーラップ
        #nperseg =  50000# 窓長さ
        #noverlap =  47500  # 窓のオーバーラップ
        #nperseg =  50000# 窓長さ   for 10Hz MECH
        #noverlap =  48750  # 窓のオーバーラップ
        #noverlap =  49750  # 窓のオーバーラップ
        nperseg =  5000# 窓長さ   for 10Hz MECH
        noverlap =  4750  # 窓のオーバーラップ
        #nperseg = 2500  # 窓長さ   for 40Hz MECH
        #noverlap = 2250  # 窓のオーバーラップ
        frequencies, times, Sxx = sig.spectrogram(filtered_signal, fs, nperseg=nperseg, noverlap=noverlap)

        # スペクトログラムのプロット
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times + start_time, frequencies, 10 * np.log10(Sxx), cmap='jet')#, vmin=-100, vmax=-70)#shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        #plt.ylim([0, 80000])  # 表示範囲を0から10kHzまでに設定
        plt.xlim(5.1,5.5)
        plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.yscale('log')
        filename = 'Spectgram_BS_%s_ch%d_SN%d.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.savefig(filename)
        plt.show()

        '''
        # スケールの範囲を設定（必要な周波数帯域に応じて調整）
        scales = np.arange(1, 5000)  # スケール範囲を広げて低周波数成分もカバー

        # チャンクサイズとオーバーラップを設定
        chunk_size = 100000  # 例: 10万サンプルのチャンク
        overlap = 50000  # 例: 5万サンプルのオーバーラップ

        # 結果を保存するリストを初期化
        coef_list = []
        time_list = []

        # チャンク数を計算
        total_chunks = (len(filtered_signal) - overlap) // (chunk_size - overlap) + 1

        # 処理開始時間
        total_start_time = time.time()

        # チャンクごとに処理
        for i, start in enumerate(range(0, len(filtered_signal), chunk_size - overlap)):
            end = min(start + chunk_size, len(filtered_signal))
            chunk_signal = filtered_signal[start:end]
            chunk_time = filtered_time[start:end]

            # チャンク処理開始時間
            start_process_time = time.time()

            # 連続ウェーブレット変換 (CWT)
            coef, freqs = pywt.cwt(chunk_signal, scales, 'morl', sampling_period=1 / fs)

            # 結果をリストに追加
            coef_list.append(coef)
            time_list.append(chunk_time)

            # チャンク処理終了時間
            end_process_time = time.time()

            # 進捗状況と処理時間をコンソールに表示
            process_time = end_process_time - start_process_time
            elapsed_time = end_process_time - total_start_time
            remaining_chunks = total_chunks - (i + 1)
            estimated_total_time = elapsed_time / (i + 1) * total_chunks
            estimated_remaining_time = estimated_total_time - elapsed_time
            progress_percent = ((i + 1) / total_chunks) * 100

            print(f"Chunk {i + 1}/{total_chunks} processed in {process_time:.2f} seconds")
            print(
                f"Elapsed time: {elapsed_time:.2f} seconds, Estimated remaining time: {estimated_remaining_time:.2f} seconds, Progress: {progress_percent:.2f}%")

        # 結果を結合
        coef_combined = np.hstack(coef_list)
        time_combined = np.hstack(time_list)

        # スペクトログラムのプロット
        plt.figure(figsize=(12, 6))
        plt.contourf(time_combined, freqs, np.abs(coef_combined), extend='both', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform (5.1s-5.6s)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim([0, 10000])  # 表示範囲を0から10kHzまでに設定
        plt.show()
        '''

        # フーリエ変換
        fft_result = fft(filtered_signal)

        # 複素振幅の計算
        complex_amplitude = ifft(fft_result)

        # 複素振幅の絶対値を計算（複素振幅強度）
        amplitude_strength = np.abs(complex_amplitude)

        # バターワースバンドパスフィルタの設計
        #lowcut = 90000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        #lowcut = 5000#1000  # カットオフ周波数 1kHz
        #highcut = 30000#10000  # カットオフ周波数 10kHz
        #lowcut = 50000#50000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        lowcut = 10000  # カットオフ周波数 1kHz
        highcut = 20000  # カットオフ周波数 10kHz
        #lowcut = 500  # カットオフ周波数 1kHz
        #lowcut = 1000  # カットオフ周波数 1kHz
        #highcut = 2700  # カットオフ周波数 10kHz
        #highcut = 10000  # カットオフ周波数 10kHz
        order = 2#3#5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        #b, a = butter(order, low, btype='low', analog=False)

        # 複素振幅強度にフィルタを適用
        filtered_amplitude_strength = np.abs(filtfilt(b, a, amplitude_strength))

        if isBSTwoBand:
            lowcut_2 = 50000#50000#1000  # カットオフ周波数 1kHz
            highcut_2 = 100000#10000  # カットオフ周波数 10kHz
            low_2 = lowcut_2 / nyquist
            high_2 = highcut_2 / nyquist
            b_2, a_2 = butter(order, [low_2, high_2], btype='band')

            # 複素振幅強度にフィルタを適用
            filtered_amplitude_strength_2 = np.abs(filtfilt(b_2, a_2, amplitude_strength))




        idx_time_st = find_closest(timedata_BS, t_st)
        idx_time_ed = find_closest(timedata_BS, t_ed)+1
        idx_time_period = find_closest(timedata_BS, t_st + 1/mod_freq) - idx_time_st + 1
        #i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        i_wave = int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        idx_time_period_Sxx = int(len(times-21)/i_wave)

        #Sxx_condAV = Sxx[:,:-1].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        # --- サイズ整合チェック & 自動 period 算出 ---
        Sxx_work = Sxx[:, :-1]  # Nyquist 行を除く（:-1 は現行コードと同じ）
        n_wave = i_wave  # ここでは 28 のはず
        n_period = Sxx_work.shape[1] // n_wave  # 割り切れる範囲の完全 period 数

        # 割り切れず余る列があれば丸ごと捨てる
        Sxx_work = Sxx_work[:, :n_wave * n_period]

        # 必要なら安全確認
        assert Sxx_work.size == len(frequencies) * n_wave * n_period, \
            f'shape mismatch: {Sxx_work.size} vs {len(frequencies) * n_wave * n_period}'

        #print('frequencies  :', len(frequencies))  # 2501
        #print('i_wave       :', n_wave)  # 28
        #print('calculated Np:', n_period)  # 257 など
        #print('Sxx_work shape:', Sxx_work.shape)  # (2501, 28*n_period)
        # reshape → 波番号平均
        Sxx_condAV = Sxx_work.reshape(len(frequencies), n_wave, n_period).mean(axis=1)
        #Sxx_condAV = Sxx[:,:-11].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        #Sxx_condAV = Sxx[:,:-21].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        # スペクトログラムのプロット
        #plt.figure(figsize=(4, 6*8.4/9), dpi=150)
        plt.figure(figsize=(4.1, 6), dpi=150)
        #plt.pcolormesh(times[:idx_time_period_Sxx]-0.01, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-120, vmax=-105)#shading='gouraud')
        #plt.pcolormesh(times[:idx_time_period_Sxx]*1000, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-120, vmax=-105)#shading='gouraud') #preset
        plt.pcolormesh(times[:idx_time_period_Sxx]*1000, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-110, vmax=-80)#shading='gouraud')
        cbar = plt.colorbar(pad=0.02)  # <- カラーバーオブジェクトを変数に格納
        cbar.set_label("Intensity [dB]", fontsize=20)  # <- ラベル設定
        cbar.ax.tick_params(labelsize=16)  # <- 目盛りフォントサイズ設定
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [ms]', fontsize=20)
        plt.ylabel('Frequency [kHz]', fontsize=20)
        #plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.ylim([0, 100])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([30, 500000])  # 表示範囲を0から10kHzまでに設定
        #plt.yscale('log')
        #plt.xlim(0,)
        plt.xlim(0,1e3/mod_freq)
        plt.tick_params(labelsize=16)
        #plt.xticks([0, 20, 40, 60, 80, 100])
        filename = 'Spectgram_condAV_BS_%s_ch%d_SN%d_v4.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

        #data_BS_condAV = filtered_amplitude_strength[idx_time_st:idx_time_ed].reshape(i_wave, -1).T.mean(axis=-1)
        data_BS_condAV = filtered_amplitude_strength.reshape(i_wave, -1).T.mean(axis=-1)
        #data_BS_condAV = amplitude_strength.reshape(i_wave, -1).T.mean(axis=-1)
        fs = 2e3#2e2#2e3#5e3#5e3#1e3
        dt = timedata_BS[1]-timedata_BS[0]
        freq = fftpack.fftfreq(len(timedata_BS[:idx_time_period-1]), dt)
        yf = fftpack.rfft(data_BS_condAV) / (int(len(data_BS_condAV) / 2))
        yf2 = np.copy(yf)
        yf2[(freq > fs)] = 0
        yf2[(freq < 0)] = 0
        y2 = np.real(fftpack.irfft(yf2) * (int(len(data_BS_condAV) / 2)))

        # 最大値とその時刻を取得
        max_value = np.max(y2)
        max_index = np.argmax(y2)
        time = timedata_BS[:idx_time_period-1]-timedata_BS[0]
        max_time = time[max_index]

        print(f"Max value: {max_value} \nMax time: {max_time}")

        if isBSTwoBand:
            data_BS_condAV_2 = filtered_amplitude_strength_2.reshape(i_wave, -1).T.mean(axis=-1)
            yf_2 = fftpack.rfft(data_BS_condAV_2) / (int(len(data_BS_condAV_2) / 2))
            yf2_2 = np.copy(yf_2)
            yf2_2[(freq > fs)] = 0
            yf2_2[(freq < 0)] = 0
            y2_2 = np.real(fftpack.irfft(yf2_2) * (int(len(data_BS_condAV_2) / 2)))

            # 最大値とその時刻を取得
            max_value_2 = np.max(y2_2)
            max_index_2 = np.argmax(y2_2)
            max_time_2 = time[max_index_2]

            print(f"Max value 2:{max_value_2} \nMax time 2: {max_time_2}")
            print(f"Ratio of Max value:{max_value/max_value_2}")

        if isFitting:
            # 時刻0から0.025sまでのデータを選択
            start_time = 0.035#0
            end_time = 0.06#0.1#0.007#0.025
            timedata_condAV = timedata_BS[:idx_time_period-1]-timedata_BS[0]
            mask = (timedata_condAV >= start_time) & (timedata_condAV <= end_time)
            filtered_timedata = timedata_condAV[mask]
            filtered_data = y2[mask]

            # 0.003sから0.005sのデータを除外
            exclude_start_time = 0.0029
            exclude_end_time = 0.0051
            exclude_mask = (filtered_timedata >= exclude_start_time) & (filtered_timedata <= exclude_end_time)
            final_timedata = filtered_timedata[~exclude_mask]
            final_data = filtered_data[~exclude_mask]

            # フィッティング
            initial_guess = [0, 0.001, 0.04, 0.0005]  # 初期パラメータの推定値
            #initial_guess = [0, 0.001, 0.020, 0.0005]  # 初期パラメータの推定値
            params, covariance = curve_fit(extreme_func, final_timedata, final_data, p0=initial_guess)
            perr = np.sqrt(np.diag(covariance))
            #perr =np.diag(params)

            # フィッティング結果のパラメータ
            y0_fit, A_fit, xc_fit, w_fit = params
            y0_fit_err, A_fit_err, xc_fit_err, w_fit_err = perr
            print(f"Fitted parameters for Extreme: \ny0 = {y0_fit}±{y0_fit_err}, \nA = {A_fit}±{A_fit_err}, \nxc = {xc_fit}±{xc_fit_err}, \nw = {w_fit}±{w_fit_err}")
            # フィッティング曲線を生成
            fit_curve = extreme_func(timedata_condAV, *params)

            # 既に同じショット番号が存在するかチェック
            existing_shots = set()
            file_path = "fitting_parameter_Extreme_50to100kHz_20240509.csv"
            header = ['ShotNo', 'reff_a99_highK', 'y0', 'y0_error', 'A','A_error', 'xc', 'xc_error', 'w',  'w_error']
            # 既存データの読み込み
            data_rows = []
            if os.path.exists(file_path):
                with open(file_path, mode='r', newline='') as file:
                    reader = csv.reader(file)
                    data_rows = list(reader)

            # ヘッダーが存在しない場合は追加
            if not data_rows:
                data_rows.append(header)

            data_name_highK = 'mwrm_highK_Power3'  # RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
            reff_a99_highK = data.getValData('reff/a99')
            reff_a99_highK_av = reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1)
            # 同じショット番号のデータが存在するかチェックし、存在する場合は上書き
            new_row = [self.ShotNo, reff_a99_highK_av, y0_fit, y0_fit_err, A_fit, A_fit_err, xc_fit,  xc_fit_err, w_fit, w_fit_err]
            shot_exists = False
            for i, row in enumerate(data_rows):
                if row[0] == str(self.ShotNo):
                    data_rows[i] = new_row
                    shot_exists = True
                    break

            if not shot_exists:
                data_rows.append(new_row)

            # ファイルに書き込む
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data_rows)

            print(f"Data for shot number {self.ShotNo} has been written to the file.")


        plt.plot(filtered_time, filtered_amplitude_strength)
        #plt.plot(filtered_time, amplitude_strength)
        #plt.plot(timedata_BS, voltdata_BS_Q)
        #plt.plot(phase_BS)
        #plt.xlim(5,5.0001)
        #plt.xlim(5,5.1)
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.show()

        fig = plt.figure(figsize=(6,6), dpi=150)
        if isFitting:
            plt.subplots_adjust(top=0.75)
        plt.plot(timedata_BS[:idx_time_period-1]-timedata_BS[0], data_BS_condAV)
        plt.plot(timedata_BS[:idx_time_period-1]-timedata_BS[0], y2)
        plt.plot(max_time, max_value, 'ro', label='Max point')
        if isBSTwoBand:
            plt.subplots_adjust(top=0.7)
            #plt.plot(timedata_BS[:idx_time_period-1]-timedata_BS[0], data_BS_condAV_2)
            plt.plot(timedata_BS[:idx_time_period-1]-timedata_BS[0], y2_2)
            plt.plot(max_time_2, max_value_2, 'go', label='Max point 2')
        title = 'Cond. Av., %s(ch.:%d), #%d, t=%.3f-%.3fs, \nBnad-pass: %.1f-%.1fkHz' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
        plt.xlabel('Time[sec]')
        if isFitting:
            plt.plot(timedata_condAV, fit_curve, 'r-', linewidth=2, label='Extreme fit')
            title_2 = f"\nFitted parameters for Extreme: \ny0 = {y0_fit}±{y0_fit_err}, \nA = {A_fit}±{A_fit_err}, \nxc = {xc_fit}±{xc_fit_err}, \nw = {w_fit}±{w_fit_err}"
            plt.title(title + title_2, fontsize=13, loc='right')
            plt.legend()
        if isBSTwoBand:
            data_name_highK = 'mwrm_highK_Power3'  # RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
            reff_a99_highK = data.getValData('reff/a99')
            timedata = data.getDimData('Time')
            idx_time_st_highK = find_closest(timedata, t_st)
            idx_time_ed_highK = find_closest(timedata, t_ed)  # +1
            rho_highK = np.mean(reff_a99_highK[idx_time_st_highK:idx_time_ed_highK], axis=0)
            title_2 = f"\nr/a={rho_highK}\nMax value:{max_value} \nMax time: {max_time}\nBand-pass 2: {lowcut_2/1e3}-{highcut_2/1e3}kHz\nMax value 2:{max_value_2} \nMax time 2: {max_time_2}\nRatio of Max value:{max_value/max_value_2}"
            title = title + title_2
            plt.title(title,loc="left")

        else:
            plt.title(title)

        #plt.xlim(0,0.01)
        plt.xlim(0, 1/mod_freq)
        #plt.xlim(0, 0.1)
        #plt.ylim(0,0.0005)
        plt.ylim(0,)
        filename = 'CondAv_BS_%s_ch%d_SN%d_t%dto%dms_Bnadpass_%.1fto%.1fkHz_fitting.png' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
        if isBSTwoBand:
            filename = 'CondAv_BS_%s_ch%d_SN%d_t%dto%dms_Bnadpass_%.1fto%.1fkHz_and_%.1fto%.1fkHz.png' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3, lowcut_2/1e3, highcut_2/1e3)
        plt.savefig(filename)
        plt.show()
        if isSaveData:
            t_data_BS = np.vstack((timedata_BS[:idx_time_period-1]-timedata_BS[0], y2))
            filename = 'CondAv_BS_%s_ch%d_SN%d_t%dto%dms_Bnadpass_%.1fto%.1fkHz.txt' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
            np.savetxt('time_' + filename, t_data_BS.T)

        return timedata_BS[:idx_time_period-1]-timedata_BS[0], y2

    def conditional_average_BSADC(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None):
        #ch1(4.0m)
        ch_I = 17
        ch_Q = 18
        #ch2(4.4m)
        #ch_I = 23
        #ch_Q = 24
        #ch3
        #ch_I = 19
        #ch_Q = 20
        #ch4(140G,4.4m)
        #ch_I = 21
        #ch_Q = 22
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        #ch_I = 9
        #data_name_BS = 'MWRM-PXI'
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_BS, start_time)
        end_index = np.searchsorted(timedata_BS, end_time)
        filtered_time = timedata_BS[start_index:end_index]
        filtered_signal = voltdata_BS_I[start_index:end_index]


        # スペクトログラムの計算
        nperseg =  5000# 窓長さ   for 10Hz MECH
        noverlap =  4750  # 窓のオーバーラップ
        nperseg =  12500# 窓長さ
        noverlap =  512  # 窓のオーバーラップ
        #nperseg =  2**15# 窓長さ
        #noverlap =  2**8  # 窓のオーバーラップ
        #nperseg =  50000# 窓長さ
        #noverlap =  47500  # 窓のオーバーラップ
        #nperseg =  50000# 窓長さ   for 10Hz MECH
        #noverlap =  48750  # 窓のオーバーラップ
        #noverlap =  49750  # 窓のオーバーラップ
        #nperseg = 2500  # 窓長さ   for 40Hz MECH
        #noverlap = 2250  # 窓のオーバーラップ
        nperseg = 2048  # ≈ 0.8 ms
        noverlap = 1536  # 75% overlap
        frequencies, times_Sxx, Sxx = sig.spectrogram(filtered_signal, fs, nperseg=nperseg, noverlap=noverlap)
        times_Sxx += t_st

        # フーリエ変換
        fft_result = fft(filtered_signal)

        # 複素振幅の計算
        complex_amplitude = ifft(fft_result)

        # 複素振幅の絶対値を計算（複素振幅強度）
        amplitude_strength = np.abs(complex_amplitude)

        # バターワースバンドパスフィルタの設計
        #lowcut = 90000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        #lowcut = 5000#1000  # カットオフ周波数 1kHz
        #highcut = 30000#10000  # カットオフ周波数 10kHz
        #lowcut = 50000#50000#1000  # カットオフ周波数 1kHz
        #highcut = 100000#10000  # カットオフ周波数 10kHz
        lowcut = 10000  # カットオフ周波数 1kHz
        highcut = 20000  # カットオフ周波数 10kHz
        #lowcut = 500  # カットオフ周波数 1kHz
        #lowcut = 1000  # カットオフ周波数 1kHz
        #highcut = 2700  # カットオフ周波数 10kHz
        #highcut = 10000  # カットオフ周波数 10kHz
        order = 2#3#5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        #b, a = butter(order, low, btype='low', analog=False)

        # 複素振幅強度にフィルタを適用
        filtered_amplitude_strength = np.abs(filtfilt(b, a, amplitude_strength))


        #Sxx_condAV = Sxx[:,:-1].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        # --- サイズ整合チェック & 自動 period 算出 ---
        Sxx_work = Sxx[:, :-1]  # Nyquist 行を除く（:-1 は現行コードと同じ）

        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        t = data.getDimData('Time')
        t = timedata_BS

        data_highK = data.getValData('Amplitude (20-200kHz)')
        data_highK = voltdata_BS_I
        plt.plot(t, data_highK)
        plt.show()

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        r = data_radhinfo.getValData('rho')

        i_ch = 13#20#14#13#9
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        voltdata_array = np.zeros((len(timedata), 32))
        for i in range(32):
            voltdata_array[:, i] = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()

        data_mp = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_mp_2d = data_mp.getValData(0)
        t_mp = data_mp.getDimData('TIME')

        _, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch, isPlot=False)
        #_, _, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch)


        #timedata_peaks_ltd = timedata_peaks_ltd_
        #arr_peak_selection = [0,2,3,5,6,7]#169714
        #arr_peak_selection = [0,6,7,8,9,10,12,13]#169715
        #arr_peak_selection = [1,2,5,6,7,8]#169712
        arr_peak_selection =[0,1,2,3,4,10,11] #169710
        #arr_peak_selection =[2,3,4] #169692
        #arr_peak_selection =[1,2,8] #169693
        #arr_peak_selection =[5,7,10] #169698
        #arr_peak_selection = [10] #169699
        #arr_peak_selection = [3,4,10] #169707
        #arr_peak_selection = [3,4,5,8,10,11,13] #169708
        timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        arr_toff = [-2e-4, 0, -2e-4, -3e-4, -3e-4, -2e-4, -1e-4]    #169710
        #arr_toff = [-2e-4, -3.5e-4, -3.5e-4, -1.5e-4, -2.5e-4, 0.5e-4]    #169712
        #arr_toff = 1e-4*np.array([-3,-1,0,-0.5,-1,-3,-2,-0.5])    #169715
        #arr_toff = 1e-4*np.array([-1.2,0,-1,-0.5,-1,-1])    #169714
        #arr_toff = 1e-4*np.array([-1,-3.5,-4])    #169692
        #arr_toff = 1e-4*np.array([-3,-2,-2])    #169693
        #arr_toff = 1e-4*np.array([-2,-1,0])    #169698
        #arr_toff = 1e-4*np.array([-0.5])    #169699
        #arr_toff = 1e-4*np.array([1.2,-2.2,-1.0])    #169707
        #arr_toff = 1e-4*np.array([-0.6,-0.9,0.1,-1,-2,-3.2,0])    #169708
        #timedata_peaks_ltd = np.array(timedata_peaks_ltd, dtype=np.float64)
        #arr_toff = float(arr_toff)
        timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 2e-3
        t_ed_condAV = 4e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3), 0]
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-2e-3):find_closest(timedata, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-2e-3):find_closest(t, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-2e-3):find_closest(t_mp, timedata_peaks_ltd[i]+4e-3)] - timedata_peaks_ltd[i]
                '''
                buf_radh = voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_highK = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                buf_mp = data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                buf_Sxx = Sxx_work[:, find_closest(times_Sxx, timedata_peaks_ltd[i]-t_st_condAV):find_closest(times_Sxx, timedata_peaks_ltd[i]+t_ed_condAV)]
                buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                buf_t_Sxx = times_Sxx[find_closest(times_Sxx, timedata_peaks_ltd[i]-t_st_condAV):find_closest(times_Sxx, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

            else:
                try:
                    buf_radh += voltdata_array[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_highK += data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ = data_highK[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)]
                    #buf_highK_ -= np.average(buf_highK_[:100], axis=0)
                    #buf_highK += buf_highK_
                    buf_mp += data_mp_2d[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV), 0]
                    buf_t_radh = timedata[find_closest(timedata, timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_highK = t[find_closest(t, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_t_mp = t_mp[find_closest(t_mp, timedata_peaks_ltd[i]-t_st_condAV):find_closest(t_mp, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]
                    buf_Sxx += Sxx_work[:,
                              find_closest(times_Sxx, timedata_peaks_ltd[i] - t_st_condAV):find_closest(times_Sxx,
                                                                                                        timedata_peaks_ltd[
                                                                                                            i] + t_ed_condAV)]
                    buf_t_Sxx = times_Sxx[
                                find_closest(times_Sxx, timedata_peaks_ltd[i] - t_st_condAV):find_closest(times_Sxx,
                                                                                                          timedata_peaks_ltd[
                                                                                                              i] + t_ed_condAV)] - \
                                timedata_peaks_ltd[i]
                except Exception as e:
                    print(e)

            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.set_axisbelow(True)
            ax2.set_axisbelow(True)
            plt.rcParams['axes.axisbelow'] = True
            b = ax2.plot(buf_t_mp, buf_mp, 'g-', label='mp_raw', zorder=1)
            label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
            c = ax1.plot( buf_t_radh, buf_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
            a = ax2.plot(buf_t_highK, 1e3*buf_highK, 'b-', label='highK', zorder=3)
            plt.title('#%d' % self.ShotNo, loc='right')
            ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
            ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            #ax1.legend((a, b, *c), ('highK', 'mp_raw', 'radh'), loc='upper left')
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
            ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
            fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            plt.show()
        '''

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_highK = buf_highK/len(timedata_peaks_ltd)
        cond_av_mp = buf_mp/len(timedata_peaks_ltd)
        cond_av_Sxx = buf_Sxx/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        #b = ax2.plot(buf_t_mp, cond_av_mp, 'g-', label='mp_raw', zorder=1)
        label_radh = "ECE(rho=" + str(r[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        a = ax2.plot(buf_t_highK, 1e3*cond_av_highK, 'b-', label='highK', zorder=3)
        plt.title('#%d' % self.ShotNo, loc='right')
        #ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', visible=True, which='major', color='#666666', linestyle='-')
        #ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax1.grid(axis='x', visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.show()

        plt.figure(figsize=(4.1, 6), dpi=150)
        #plt.pcolormesh(times[:idx_time_period_Sxx]-0.01, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-120, vmax=-105)#shading='gouraud')
        #plt.pcolormesh(times[:idx_time_period_Sxx]*1000, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-120, vmax=-105)#shading='gouraud') #preset
        plt.pcolormesh(buf_t_Sxx, frequencies/1e3, 10 * np.log10(cond_av_Sxx), cmap='jet')#, vmin=-110, vmax=-80)#shading='gouraud')
        cbar = plt.colorbar(pad=0.02)  # <- カラーバーオブジェクトを変数に格納
        cbar.set_label("Intensity [dB]", fontsize=20)  # <- ラベル設定
        cbar.ax.tick_params(labelsize=16)  # <- 目盛りフォントサイズ設定
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [ms]', fontsize=20)
        plt.ylabel('Frequency [kHz]', fontsize=20)
        #plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.ylim([0, 100])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([30, 500000])  # 表示範囲を0から10kHzまでに設定
        #plt.yscale('log')
        #plt.xlim(0,)
        #plt.xlim(0,1e3/mod_freq)
        plt.tick_params(labelsize=16)
        #plt.xticks([0, 20, 40, 60, 80, 100])
        filename = 'Spectgram_condAV_BS_%s_ch%d_SN%d_ITBcollapse.png' % (data_name_BS, ch_I, self.ShotNo)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

        return buf_highK, buf_mp, buf_radh, buf_t_highK, buf_t_mp, buf_t_radh, timedata_peaks_ltd, r, i_ch

    def calc_stats_cross_correlation_BSADC_condAV_MECH(self, t_st, t_ed, mod_freq, isCondAV=True):
        num_loop = np.int((t_ed-t_st)*mod_freq)
        array_max_timing=[]
        array_max_cc=[]
        array_lag_at_max_cc=[]
        for i in range(num_loop):
            #max_timing, max_cc, lag_at_max_cc = self.cross_correlation_BSADC_condAV_MECH(t_st=Decimal(str(t_st+i/mod_freq)), t_ed=Decimal(str(t_st+(i+1)/mod_freq)), mod_freq=mod_freq)
            max_timing, max_cc, lag_at_max_cc = self.cross_correlation_BSADC_condAV_MECH(t_st=round(t_st+i/mod_freq, 3), t_ed=round(t_st+(i+1)/mod_freq,3), mod_freq=mod_freq, isPlot=False)
            if max_timing < 6e-3:
                array_max_timing.append(max_timing)
                array_max_cc.append(max_cc)
                array_lag_at_max_cc.append(lag_at_max_cc)
            else:
                print(f'Skip data because max timing is {max_timing:.6e}')
        print(f'Max timing:     mean={np.mean(array_max_timing):.6e}, stdev={np.std(array_max_timing):.6e}')
        print(f'Max CC:         mean={np.mean(array_max_cc):.6e}, stdev={np.std(array_max_cc):.6e}')
        print(f'Lag at Max CC:  mean={np.mean(array_lag_at_max_cc):.6e}, stdev={np.std(array_lag_at_max_cc):.6e}')

    def cross_correlation_BSADC_condAV_MECH(self, t_st, t_ed, mod_freq, isCondAV=True, isPlot=True):
        print(t_st, t_ed)
        data_name_highK_3 = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_name_highK_2 = 'mwrm_highK_Power1' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_name_highK_1 = 'mwrm_highK_Power2' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_1 = AnaData.retrieve(data_name_highK_1, self.ShotNo, 1)
        data_2 = AnaData.retrieve(data_name_highK_2, self.ShotNo, 1)
        data_3 = AnaData.retrieve(data_name_highK_3, self.ShotNo, 1)
        t = data_1.getDimData('Time')
        reff_a99_highK_1 = data_1.getValData('reff/a99')
        reff_a99_highK_2 = data_2.getValData('reff/a99')
        reff_a99_highK_3 = data_3.getValData('reff/a99')
        idx_time_st = find_closest(t, t_st)
        idx_time_ed = find_closest(t, t_ed)
        reff_a99_highK_av_1 = reff_a99_highK_1[idx_time_st:idx_time_ed].mean(axis=-1)
        reff_a99_highK_av_2 = reff_a99_highK_2[idx_time_st:idx_time_ed].mean(axis=-1)
        reff_a99_highK_av_3 = reff_a99_highK_3[idx_time_st:idx_time_ed].mean(axis=-1)
        print(reff_a99_highK_av_1, reff_a99_highK_av_2, reff_a99_highK_av_3)
        '''
        ch_I = 3
        ch_I_2 = 1#5#19
        ch_Q = 16
        #data_name_DBS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'    #9-O port comb
        data_name_BS_2 = 'MWRM-COMB'    #9-O port comb
        #data_name_DBS = 'MWRM-PXI'      #9-O R
        #data_name_DBS = 'MWRM-PXI3'      #3-O port comb-U
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_I_2 = VoltData.retrieve(data_name_BS_2,self.ShotNo, 1,ch_I_2).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()
        timedata_BS_2 = TimeData.retrieve(data_name_BS_2, self.ShotNo, 1, ch_I_2).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        '''
        #ch1(4.0m)
        ch_I_2 = 17
        #ch_Q = 18
        #ch2(4.4m)
        #ch_I_2 = 23
        #ch_Q = 24
        #ch3
        ch_I = 19
        ch_Q = 20
        #ch4(140G,4.4m)
        #ch_I_2 = 21
        #ch_Q = 22
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_I_2 = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I_2).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()
        timedata_BS_2 = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I_2).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_BS, start_time)
        end_index = np.searchsorted(timedata_BS, end_time)
        filtered_time = timedata_BS[start_index:end_index]
        filtered_signal = voltdata_BS_I[start_index:end_index]
        start_index = np.searchsorted(timedata_BS_2, start_time)
        end_index = np.searchsorted(timedata_BS_2, end_time)
        filtered_time_2 = timedata_BS_2[start_index:end_index]
        filtered_signal_2 = voltdata_BS_I_2[start_index:end_index]

        if isCondAV:
            interval_time_probe = 1
            i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
            idx_probe_time_st = find_closest(timedata_BS, t_st)
            idx_probe_time_ed = find_closest(timedata_BS, t_ed)
            idx_probe_time_period = find_closest(timedata_BS, t_st + 1/mod_freq) - idx_probe_time_st
            filtered_signal_condAV = voltdata_BS_I[idx_probe_time_st:idx_probe_time_ed:interval_time_probe].reshape(i_wave, -1).T.mean(axis=-1)
            filtered_signal_condAV_2 = voltdata_BS_I_2[idx_probe_time_st:idx_probe_time_ed:interval_time_probe].reshape(i_wave, -1).T.mean(axis=-1)

            # サンプリング周波数（Hz）
            #fs = 48000  # 48kHzを例として使用
            duration = t_ed - t_st#1.0  # 信号の長さ（秒）
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # 例として、15kHzの正弦波とノイズを持つ信号を生成
            #x = np.sin(2 * np.pi * 15000 * t) + 0.5 * np.random.randn(len(t))
            #y = np.sin(2 * np.pi * 15000 * t + np.pi / 4) + 0.5 * np.random.randn(len(t))
            x = filtered_signal_condAV
            y = filtered_signal_condAV_2

        else:
            filtered_signal = voltdata_BS_I[start_index:end_index]
            start_index = np.searchsorted(timedata_BS_2, start_time)
            end_index = np.searchsorted(timedata_BS_2, end_time)
            filtered_time_2 = timedata_BS_2[start_index:end_index]
            filtered_signal_2 = voltdata_BS_I_2[start_index:end_index]

            x = filtered_signal
            y = filtered_signal_2

        # フィルタの設定
        lowcut = 10000.0  # 10kHz
        highcut = 20000.0  # 20kHz
        order = 3#5#3#5  # フィルタの次数

        # ウィンドウとステップサイズの設定（サンプル数）
        #window_size = int(0.05 * fs)  # 0.05秒のウィンドウ
        #step_size = int(0.01 * fs)  # 0.01秒のステップ
        window_size = int(0.0005 * fs)  # 0.05秒のウィンドウ
        step_size = int(0.0002 * fs)  # 0.01秒のステップ

        '''
        plt.plot(t, x[:-1])
        plt.plot(t, y[:-1])
        plt.show()
        '''

        # 相互相関の時間変化を計算
        times, lags, cross_correlations = compute_time_evolution_cross_correlation(
            x, y, fs, lowcut, highcut, window_size, step_size, order)

        # クロスコリレーションを2D配列として格納
        cc_matrix = np.array(cross_correlations)


        # グローバルに相互相関が最大となる箇所を探索
        max_val = np.max(cc_matrix)
        max_idx = np.unravel_index(np.argmax(cc_matrix), cc_matrix.shape)
        time_idx = max_idx[0]
        lag_idx = max_idx[1]

        # 最大相互相関が生じる時間（秒）とそのスライス取得
        time_of_max = times[time_idx]
        cc_at_time = cc_matrix[time_idx, :]

        # 2つ目のプロット: 特定の時間（6ms）の相互相関
        time_point_ms = 20#6#3.5#4.0#3.0#3#5.0  # 表示したい時間（ミリ秒）
        time_point = time_point_ms / 1000  # 秒に変換

        # 時間リストから最も近い時間のインデックスを取得
        index = np.argmin(np.abs(np.array(times) - time_point))
        cc_at_time = cc_matrix[index, :]

        if isPlot:
            # クロスコリレーション行列を画像としてプロット（横軸を時間、縦軸をラグに変更）
            fig, axes = plt.subplots(2, 1, figsize=(6.3, 10), dpi=150)
            plt.rcParams['axes.linewidth'] = 2.5
            plt.rcParams['font.weight'] = 'bold'  # 太字設定

            # クロスコリレーション行列を画像としてプロット
            #plt.figure(figsize=(8, 6))
            #extent = [lags[0], lags[-1], times[-1], times[0]]
            extent = [1e3*times[0], 1e3*times[-1], 1e3*lags[0], 1e3*lags[-1]]  # extentの順序を変更-
            im = axes[0].imshow(cc_matrix.T, aspect='auto', extent=extent, cmap='seismic',origin='lower', vmin=-0.7, vmax=0.7)
            #cb = fig.colorbar(im, ax=axes[0], label='cross-correlation', pad=0.01).set_label('cross-correlation', fontdict={'size':14})
            #cb.axes[0].tick_params(labelsize=16)
            cb = fig.colorbar(im, ax=axes[0], pad=0.01)
            cb.set_label('Cross-correlation', fontdict={'size': 20, 'weight':1000})
            cb.ax.tick_params(labelsize=16)  # 修正: cb.axを使用
                        # 目盛りの数字のフォントを太くする
            for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
                label.set_fontweight('bold')  # 'bold'で太字化
                label.set_fontsize(14)  # フォントサイズも設定可能
            axes[0].spines["top"].set_linewidth(2.5)
            axes[0].spines["left"].set_linewidth(2.5)
            axes[0].spines["bottom"].set_linewidth(2.5)
            axes[0].spines["right"].set_linewidth(2.5)
            axes[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=16, width=2.5)
            axes[0].grid(axis='x', which="major")
            axes[0].grid(axis='y')
            axes[0].set_ylabel('Lag [ms]', fontsize=20, fontweight=1000)
            axes[0].set_xlabel('Time [ms]', fontsize=20, fontweight=1000)
            #title = '%s(ch.:%d & %d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I,ch_I_2, self.ShotNo, t_st, t_ed)
            title = 'Cond. AV. %s(ch.:%d & %d), #%d, t=%.4f-%.4fs\n[reff/a99 of BS]Ch.1:%.3f, Ch.2:%.3f, Ch.3:%.3f' % (data_name_BS, ch_I,ch_I_2, self.ShotNo, t_st, t_ed, reff_a99_highK_av_1, reff_a99_highK_av_2, reff_a99_highK_av_3)
            axes[0].set_title(title, fontsize=14)
            #axes[0].set_xlim(0,0.02)
            #axes[0].set_xlim(0,0.025)
            axes[0].set_xlim(0,1e3/mod_freq)
            # 最大相互相関発生時の縦線
            axes[0].axvline(x=1e3*time_of_max, color='black', linestyle='--', linewidth=2)


            axes[1].plot(lags, cc_at_time)
            axes[1].set_xlabel('Lag [s]', fontsize=14, fontweight=3.5)
            axes[1].set_ylabel('Cross-correlation', fontsize=14)
            #title = 'Cond. AV. %s(ch.:%d & %d), #%d, t=%.4f s' % (data_name_BS, ch_I,ch_I_2, self.ShotNo, times[index])
            title = 'Cond. AV. %s(ch.:%d & %d), #%d, Max timing:t=%.4f s\n[reff/a99 of BS]Ch.1:%.3f, Ch.2:%.3f, Ch.3:%.3f' % (data_name_BS, ch_I,ch_I_2, self.ShotNo, time_of_max, reff_a99_highK_av_1, reff_a99_highK_av_2, reff_a99_highK_av_3)
            #axes[1].set_title(f'Cross-correlation of t={times[index] * 1000:.1f} ms')
            axes[1].set_title(title, fontsize=14)
            axes[1].grid(True)
            plt.tight_layout()
            #filename = 'Cross_Correlation_%s_ch%dto%d_SN%d_%.3f_%.3fs.png' % (data_name_DBS, ch_I, ch_I_2, self.ShotNo, t_st, t_ed)
            #filename = 'Cross_Correlation_%s_ch%dto%d_SN%d_%.3f_%.3fs_condAV_maxcor.png' % (data_name_BS, ch_I, ch_I_2, self.ShotNo, t_st, t_ed)
            #filename = 'Cross_Correlation_%s_ch%dto%d_SN%d_%.3f_%.3fs_maxcor.png' % (data_name_BS, ch_I, ch_I_2, self.ShotNo, t_st, t_ed)
            filename = 'Cross_Correlation_%s_ch%dto%d_SN%d_%.3f_%.3fs_maxcor_t20ms.png' % (data_name_BS, ch_I, ch_I_2, self.ShotNo, t_st, t_ed)
            plt.savefig(filename)
            plt.show()
            plt.close()
            plt.clf()
            #filename = 'Cross_Correlation_%s_ch%dto%d_t%.4fs_SN%d_%.3f_%.3fs_maxcor.txt' % (data_name_BS, ch_I, ch_I_2, time_of_max, self.ShotNo, t_st, t_ed)
            filename = 'Cross_Correlation_%s_ch%dto%d_t%.4fs_SN%d_%.3f_%.3fs_maxcor.txt' % (data_name_BS, ch_I, ch_I_2,time_point, self.ShotNo, t_st, t_ed)
            np.savetxt(filename, np.c_[lags, cc_at_time])
        print(time_of_max, np.max(cc_at_time), lags[np.argmax(cc_at_time)])
        return time_of_max, np.max(cc_at_time), lags[np.argmax(cc_at_time)]

    def cross_correlation_BSADC(self, t_st, t_ed):
        '''
        ch_I = 3
        ch_I_2 = 19#5
        ch_Q = 16
        #data_name_DBS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'    #9-O port comb
        #data_name_DBS = 'MWRM-PXI'      #9-O R
        #data_name_DBS = 'MWRM-PXI3'      #3-O port comb-U
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_I_2 = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I_2).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()
        timedata_BS_2 = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I_2).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        '''
        #ch1(4.0m)
        ch_I_2 = 17
        #ch_Q = 18
        #ch2(4.4m)
        #ch_I_2 = 23
        #ch_Q = 24
        #ch3
        ch_I = 19
        ch_Q = 20
        #ch4(140G,4.4m)
        #ch_I_2 = 21
        #ch_Q = 22
        #data_name_BS = 'MWRM-COMB'    #3-O port comb-R
        data_name_BS = 'MWRM-COMB2'
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_I = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I).get_val()
        voltdata_BS_I_2 = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_I_2).get_val()
        voltdata_BS_Q = VoltData.retrieve(data_name_BS,self.ShotNo, 1,ch_Q).get_val()
        timedata_BS = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I).get_val()
        timedata_BS_2 = TimeData.retrieve(data_name_BS, self.ShotNo, 1, ch_I_2).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_BS[1] - timedata_BS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_BS, start_time)
        end_index = np.searchsorted(timedata_BS, end_time)
        filtered_time = timedata_BS[start_index:end_index]
        filtered_signal = voltdata_BS_I[start_index:end_index]
        start_index = np.searchsorted(timedata_BS_2, start_time)
        end_index = np.searchsorted(timedata_BS_2, end_time)
        filtered_time_2 = timedata_BS_2[start_index:end_index]
        filtered_signal_2 = voltdata_BS_I_2[start_index:end_index]

        # サンプリング周波数（Hz）
        #fs = 48000  # 48kHzを例として使用
        duration = t_ed - t_st#1.0  # 信号の長さ（秒）
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # 例として、15kHzの正弦波とノイズを持つ信号を生成
        #x = np.sin(2 * np.pi * 15000 * t) + 0.5 * np.random.randn(len(t))
        #y = np.sin(2 * np.pi * 15000 * t + np.pi / 4) + 0.5 * np.random.randn(len(t))
        x = filtered_signal
        y = filtered_signal_2

        # フィルタの設定
        lowcut = 10000.0  # 10kHz
        highcut = 20000.0  # 20kHz
        order = 3#5#3#5  # フィルタの次数

        # ウィンドウとステップサイズの設定（サンプル数）
        #window_size = int(0.05 * fs)  # 0.05秒のウィンドウ
        #step_size = int(0.01 * fs)  # 0.01秒のステップ
        window_size = int(0.0005 * fs)  # 0.05秒のウィンドウ
        step_size = int(0.0002 * fs)  # 0.01秒のステップ

        '''
        plt.plot(t, x[:-1])
        plt.plot(t, y[:-1])
        plt.show()
        '''

        # 相互相関の時間変化を計算
        times, lags, cross_correlations = compute_time_evolution_cross_correlation(
            x, y, fs, lowcut, highcut, window_size, step_size, order)

        # クロスコリレーションを2D配列として格納
        cc_matrix = np.array(cross_correlations)

        # クロスコリレーション行列を画像としてプロット（横軸を時間、縦軸をラグに変更）
        fig, axes = plt.subplots(2, 1, figsize=(9, 15))

        # クロスコリレーション行列を画像としてプロット
        #plt.figure(figsize=(8, 6))
        #extent = [lags[0], lags[-1], times[-1], times[0]]
        extent = [times[0], times[-1], lags[0], lags[-1]]  # extentの順序を変更-
        im = axes[0].imshow(cc_matrix.T, aspect='auto', extent=extent, cmap='seismic',origin='lower', vmin=-0.8, vmax=0.8)
        fig.colorbar(im, ax=axes[0], label='cross-correlation').set_label('cross-correlation', fontdict={'size':14})
        axes[0].grid(axis='x', which="major")
        axes[0].grid(axis='y')
        axes[0].set_ylabel('Lag [s]', fontsize=14)
        axes[0].set_xlabel('Time [s]', fontsize=14)
        #title = '%s(ch.:%d & %d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I,ch_I_2, self.ShotNo, t_st, t_ed)
        title = '%s(ch.:%d & %d), #%d, t=%.4f-%.4fs' % (data_name_BS, ch_I,ch_I_2, self.ShotNo, t_st, t_ed)
        axes[0].set_title(title, fontsize=14)
        axes[0].set_xlim(0,)

        # 2つ目のプロット: 特定の時間（6ms）の相互相関
        time_point_ms = 3.0#3#5.0  # 表示したい時間（ミリ秒）
        time_point = time_point_ms / 1000  # 秒に変換

        # 時間リストから最も近い時間のインデックスを取得
        index = np.argmin(np.abs(np.array(times) - time_point))
        cc_at_time = cc_matrix[index, :]

        axes[1].plot(lags, cc_at_time)
        axes[1].set_xlabel('Lag [s]')
        axes[1].set_ylabel('Cross-correlation')
        title = '%s(ch.:%d & %d), #%d, t=%.4f s' % (data_name_BS, ch_I,ch_I_2, self.ShotNo, times[index])
        #axes[1].set_title(f'Cross-correlation of t={times[index] * 1000:.1f} ms')
        axes[1].set_title(title)
        axes[1].grid(True)
        plt.tight_layout()
        #filename = 'Cross_Correlation_%s_ch%dto%d_SN%d_%.3f_%.3fs.png' % (data_name_DBS, ch_I, ch_I_2, self.ShotNo, t_st, t_ed)
        filename = 'Cross_Correlation_%s_ch%dto%d_SN%d_%.3f_%.3fs_v2.png' % (data_name_BS, ch_I, ch_I_2, self.ShotNo, t_st, t_ed)
        plt.savefig(filename)
        plt.show()

    def ana_plot_DBSADC_condAV_MECH(self, t_st, t_ed, mod_freq):
        ch_I =9
        ch_Q = 16
        #data_name_DBS = 'MWRM-COMB'    #3-O port comb-R
        #data_name_DBS = 'MWRM-COMB2'    #9-O port comb
        data_name_DBS = 'MWRM-PXI'      #9-O R
        #data_name_DBS = 'MWRM-PXI3'      #3-O port comb-U
        voltdata_DBS_I = VoltData.retrieve(data_name_DBS,self.ShotNo, 1,ch_I).get_val()
        voltdata_DBS_Q = VoltData.retrieve(data_name_DBS,self.ShotNo, 1,ch_Q).get_val()
        timedata_DBS = TimeData.retrieve(data_name_DBS, self.ShotNo, 1, ch_I).get_val()

        # サンプリング周波数の計算
        fs = 1 / (timedata_DBS[1] - timedata_DBS[0])

        # 5秒から7秒の間のデータを抽出
        start_time = t_st  # 5秒
        end_time = t_ed  # 7秒
        start_index = np.searchsorted(timedata_DBS, start_time)
        end_index = np.searchsorted(timedata_DBS, end_time)
        filtered_time = timedata_DBS[start_index:end_index]
        filtered_signal = voltdata_DBS_I[start_index:end_index]

        # フーリエ変換
        fft_result = fft(filtered_signal)
        freqs = fftfreq(len(filtered_signal), 1 / fs)

        # 周波数スペクトルのプロット
        plt.figure(figsize=(6, 6))
        plt.plot(freqs[:len(freqs) // 2], np.abs(fft_result)[:len(freqs) // 2])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.yscale('log')
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed)
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        filename = 'Freq_Spectrum_%s_ch%d_SN%d.png' % (data_name_DBS, ch_I, self.ShotNo)
        plt.savefig(filename)
        plt.show()

        # スペクトログラムの計算
        #nperseg =  12500# 窓長さ
        #noverlap =  512  # 窓のオーバーラップ
        #nperseg =  2**14# 窓長さ
        #noverlap =  2**7  # 窓のオーバーラップ
        nperseg =  50000# 窓長さ  for 10kHz MECH
        #noverlap =  48750  # 窓のオーバーラップ
        nperseg =  5000# 窓長さ  for 10kHz MECH
        noverlap =  4750# 窓のオーバーラップ
        #noverlap =  49750  # 窓のオーバーラップ
        #nperseg = 2500  # 窓長さ  for 40Hz MECH
        #noverlap = 2250  # 窓のオーバーラップ
        frequencies, times, Sxx = sig.spectrogram(filtered_signal, fs, nperseg=nperseg, noverlap=noverlap)


        # スペクトログラムのプロット
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times + start_time, frequencies, 10 * np.log10(Sxx), cmap='jet')#, vmin=-100, vmax=-70)#shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        #plt.ylim([0, 40000])  # 表示範囲を0から10kHzまでに設定
        plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.yscale('log')
        plt.xlim(5.1,5.5)
        filename = 'Spectgram_%s_ch%d_SN%d.png' % (data_name_DBS, ch_I, self.ShotNo)
        plt.savefig(filename)
        plt.show()


        # フーリエ変換
        fft_result = fft(filtered_signal)

        # 複素振幅の計算
        complex_amplitude = ifft(fft_result)

        # 複素振幅の絶対値を計算（複素振幅強度）
        amplitude_strength = np.abs(complex_amplitude)

        # バターワースバンドパスフィルタの設計
        #lowcut = 20000#1000  # カットオフ周波数 1kHz
        #highcut = 80000#10000  # カットオフ周波数 10kHz
        lowcut = 100000#1000  # カットオフ周波数 1kHz
        highcut = 500000#10000  # カットオフ周波数 10kHz
        #lowcut = 1000  # カットオフ周波数 1kHz
        #highcut = 10000  # カットオフ周波数 10kHz
        order = 3#5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        #b, a = butter(order, low, btype='low', analog=False)

        # 複素振幅強度にフィルタを適用
        filtered_amplitude_strength = np.abs(filtfilt(b, a, amplitude_strength))



        idx_time_st = find_closest(timedata_DBS, t_st)
        idx_time_ed = find_closest(timedata_DBS, t_ed)+1
        idx_time_period = find_closest(timedata_DBS, t_st + 1/mod_freq) - idx_time_st + 1
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))

        #Sxx_DBS_condAV = Sxx.reshape(i_wave).T.mean(axis=-1)

        #data_DBS_condAV = filtered_amplitude_strength[idx_time_st:idx_time_ed].reshape(i_wave, -1).T.mean(axis=-1)
        data_DBS_condAV = filtered_amplitude_strength.reshape(i_wave, -1).T.mean(axis=-1)
        #data_DBS_condAV = amplitude_strength.reshape(i_wave, -1).T.mean(axis=-1)
        fs = 1e3#5e3#5e3#1e3
        dt = timedata_DBS[1]-timedata_DBS[0]
        freq = fftpack.fftfreq(len(timedata_DBS[:idx_time_period-1]), dt)
        yf = fftpack.rfft(data_DBS_condAV) / (int(len(data_DBS_condAV) / 2))
        yf2 = np.copy(yf)
        yf2[(freq > fs)] = 0
        yf2[(freq < 0)] = 0
        y2 = np.real(fftpack.irfft(yf2) * (int(len(data_DBS_condAV) / 2)))

        plt.plot(filtered_time, filtered_amplitude_strength)
        #plt.plot(filtered_time, amplitude_strength)
        #plt.plot(timedata_DBS, voltdata_DBS_Q)
        #plt.plot(phase_DBS)
        #plt.xlim(5,5.0001)
        #plt.xlim(5,5.1)
        title = '%s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.show()

        fig = plt.figure(figsize=(6,4), dpi=150)
        plt.plot(timedata_DBS[:idx_time_period-1]-timedata_DBS[0], data_DBS_condAV)
        plt.plot(timedata_DBS[:idx_time_period-1]-timedata_DBS[0], y2)
        #plt.xlim(0,0.01)
        title = 'Cond. Av., %s(ch.:%d), #%d, t=%.3f-%.3fs, \nBnad-pass: %.1f-%.1fkHz' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
        plt.xlabel('Time[sec]')
        plt.title(title)
        plt.xlim(0, 1/mod_freq)
        filename = 'CondAv_%s_ch%d_SN%d_t%dto%dms_Bnadpass_%.1fto%.1fkHz.png' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed, lowcut/1e3, highcut/1e3)
        plt.savefig(filename)
        plt.show()

        #idx_time_period_Sxx = int(len(times-1)/i_wave)
        idx_time_period_Sxx = int(len(times-1)/i_wave)
        Sxx_condAV = Sxx[:,:-1].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        #Sxx_condAV = Sxx[:,:-11].reshape(len(frequencies), i_wave, idx_time_period_Sxx).mean(axis=1)
        # スペクトログラムのプロット
        plt.figure(figsize=(4, 6), dpi=150)
        #plt.pcolormesh(times[:idx_time_period_Sxx]-0.01, frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-65, vmax=-55)#shading='gouraud')
        plt.pcolormesh(times[:idx_time_period_Sxx], frequencies/1e3, 10 * np.log10(Sxx_condAV), cmap='jet', vmin=-100, vmax=-90)#shading='gouraud')
        plt.colorbar(label='Intensity [dB]')
        title = 'CondAV. %s(ch.:%d), #%d, t=%.4f-%.4fs' % (data_name_DBS, ch_I, self.ShotNo, t_st, t_ed)
        plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [kHz]')
        #plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([300, 100000])  # 表示範囲を0から10kHzまでに設定
        #plt.ylim([100, 500000])  # 表示範囲を0から10kHzまでに設定
        plt.ylim([0, 100])  # 表示範囲を0から10kHzまでに設定
        plt.xlim(0,)
        #plt.yscale('log')
        #plt.xlim(5.1,5.5)
        filename = 'Spectgram_condAV_%s_ch%d_SN%d_v5.png' % (data_name_DBS, ch_I, self.ShotNo)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()



    def ana_plot_highK_condAV_MECH(self, t_st, t_ed, mod_freq):
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power1' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        #data_name_highK = 'mwrm_highK_Power2' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        vnum = data.getValNo()
        timedata = data.getDimData('Time')

        data_highK = data.getValData('Amplitude (20-200kHz)')
        #data_highK = data.getValData('Amplitude (3-30kHz)')
        reff_a99_highK = data.getValData('reff/a99')

        data_name_highK_probe = 'MWRM-PXI'
        voltdata_highK_probe = VoltData.retrieve(data_name_highK_probe,self.ShotNo, 1, 4).get_val()
        timedata_highK_probe = TimeData.retrieve(data_name_highK_probe, self.ShotNo, 1, 1).get_val()
        voltdata_highK_probe_01 = np.where(voltdata_highK_probe < -0.5, 1, 0)


        interval_time_probe = 10
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#+1
        idx_time_period = find_closest(timedata, t_st + 1/mod_freq) - idx_time_st
        #i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        i_wave = int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        idx_probe_time_st = find_closest(timedata_highK_probe, t_st)
        idx_probe_time_ed = find_closest(timedata_highK_probe, t_ed)
        idx_probe_time_period = find_closest(timedata_highK_probe, t_st + 1/mod_freq) - idx_probe_time_st
        #i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        i_wave = int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        data_highK_probe_condAV = voltdata_highK_probe_01[idx_probe_time_st:idx_probe_time_ed:interval_time_probe].reshape(i_wave, -1).T.mean(axis=-1)
        data_highK_probe_condAV_01 = np.where(data_highK_probe_condAV<1, 0, 1)
        #data_highK_condAV = data_highK[idx_time_st:idx_time_ed].reshape(np.int(mod_freq*(t_ed-t_st)), -1).T.mean(axis=-1)
        data_highK_condAV = data_highK[idx_time_st:idx_time_ed].reshape(i_wave, -1).T.mean(axis=-1)
        #data_highK_condAV_mask = (data_highK[idx_time_st:idx_time_ed].reshape(i_wave, -1).T * data_highK_probe_condAV_01.T).mean(axis=-1)*i_wave/np.sum(data_highK_probe_condAV_01)
        data_highK_condAV_mask = (data_highK[idx_time_st:idx_time_ed].reshape(i_wave,
                                                                              -1).T * data_highK_probe_condAV_01.reshape(
            -1, 1)).mean(axis=-1) * i_wave / np.sum(data_highK_probe_condAV_01)

        fs = 5e3#5e3#5e3#1e3
        dt = timedata[1] - timedata[0]
        #freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        freq = fftpack.fftfreq(len(timedata[:idx_time_period]), dt)
        #yf = fftpack.rfft(data_highK_condAV_mask) / (int(len(data_highK_condAV_mask) / 2))
        yf = fftpack.rfft(data_highK_condAV) / (int(len(data_highK_condAV) / 2))
        yf2 = np.copy(yf)
        yf2[(freq > fs)] = 0
        yf2[(freq < 0)] = 0
        y2 = np.real(fftpack.irfft(yf2) * (int(len(data_highK_condAV_mask) / 2)))

        # 時刻0から0.025sまでのデータを選択
        start_time = 0
        end_time = 0.025
        timedata_condAV = timedata[:idx_time_period]-timedata[0]
        mask = (timedata_condAV >= start_time) & (timedata_condAV <= end_time)
        filtered_timedata = timedata_condAV[mask]
        filtered_data = data_highK_condAV[mask]

        # 0.003sから0.005sのデータを除外
        exclude_start_time = 0.0029
        exclude_end_time = 0.0051
        exclude_mask = (filtered_timedata >= exclude_start_time) & (filtered_timedata <= exclude_end_time)
        final_timedata = filtered_timedata[~exclude_mask]
        final_data = filtered_data[~exclude_mask]

        # フィッティング
        initial_guess = [0, 0.001, 0.005, 0.0005]  # 初期パラメータの推定値
        #initial_guess = [0, 0.001, 0.020, 0.0005]  # 初期パラメータの推定値
        params, covariance = curve_fit(extreme_func, final_timedata, final_data, p0=initial_guess)
        perr = np.sqrt(np.diag(covariance))
        #perr =np.diag(params)

        # フィッティング結果のパラメータ
        y0_fit, A_fit, xc_fit, w_fit = params
        y0_fit_err, A_fit_err, xc_fit_err, w_fit_err = perr
        print(f"Fitted parameters for Extreme: \ny0 = {y0_fit}±{y0_fit_err}, \nA = {A_fit}±{A_fit_err}, \nxc = {xc_fit}±{xc_fit_err}, \nw = {w_fit}±{w_fit_err}")
        reff_a99_highK_av = reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1)
        '''
        # 既に同じショット番号が存在するかチェック
        existing_shots = set()
        file_path = "fitting_parameter_Extreme_20240509.csv"
        header = ['ShotNo', 'reff_a99_highK', 'y0', 'y0_error', 'A','A_error', 'xc', 'xc_error', 'w',  'w_error']
        # 既存データの読み込み
        data_rows = []
        if os.path.exists(file_path):
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                data_rows = list(reader)

        # ヘッダーが存在しない場合は追加
        if not data_rows:
            data_rows.append(header)

        # 同じショット番号のデータが存在するかチェックし、存在する場合は上書き
        new_row = [self.ShotNo, reff_a99_highK_av, y0_fit, y0_fit_err, A_fit, A_fit_err, xc_fit,  xc_fit_err, w_fit, w_fit_err]
        shot_exists = False
        for i, row in enumerate(data_rows):
            if row[0] == str(self.ShotNo):
                data_rows[i] = new_row
                shot_exists = True
                break

        if not shot_exists:
            data_rows.append(new_row)

        # ファイルに書き込む
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_rows)

        print(f"Data for shot number {self.ShotNo} has been written to the file.")
        '''

        '''
        # フィッティング
        initial_guess = [0.002]  # sの初期パラメータの推定値
        params_2, covariance_2 = curve_fit(lognorm_func, final_timedata, final_data, p0=initial_guess)

        # フィッティング結果のパラメータ
        s_fit = params[0]
        print(f"Fitted parameter for LogNorm: s = {s_fit}")
        '''

        # フィッティング曲線を生成
        fit_curve = extreme_func(timedata_condAV, *params)
        #fit_curve_2 = lognorm_func(timedata_condAV, *params_2)


        fig = plt.figure(figsize=(8,6), dpi=150)
        plt.plot(timedata[:idx_time_period]-timedata[0], data_highK_condAV)#, label=label)
        plt.plot(timedata[:idx_time_period]-timedata[0], data_highK_condAV_mask, label='masked')#, label=label)
        label_lowpass = 'Low-pass(<%dHz)' % fs
        plt.plot(timedata[:idx_time_period]-timedata[0], y2, label=label_lowpass)#, label=label)
        #plt.xlim(4.4, 4.6)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs, reff/a99=%.3f' % (data_name_highK, self.ShotNo, t_st, t_ed, reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1))
        plt.title(title, fontsize=18, loc='right')
        plt.xlim(0, timedata[idx_time_period]-timedata[0])
        #plt.xlim(0,0.1)
        plt.legend()
        filename = "highK_condAV_mask_lowpass_t%dto%dms_No%d_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        #plt.savefig(filename)
        plt.show()
        '''

        fig = plt.figure(figsize=(8,6), dpi=150)
        #plt.subplots_adjust(left=0.05, right=0.995, bottom=0.05, top=0.995)
        plt.subplots_adjust(top=0.75)
        plt.plot(timedata[:idx_time_period]-timedata[0], data_highK_condAV)#, label=label)
        plt.plot(timedata[:idx_time_period]-timedata[0], data_highK_condAV_mask, label='masked')#, label=label)
        plt.plot(final_timedata, final_data, 'b-', markersize=2, label='Data')
        plt.plot(timedata_condAV, fit_curve, 'r-', linewidth=2, label='Extreme fit')
        #plt.plot(timedata_condAV, fit_curve_2, 'y-', linewidth=2, label='LogNorm fit')
        label_lowpass = 'Low-pass(<%dHz)' % fs
        plt.plot(timedata[:idx_time_period]-timedata[0], y2, label=label_lowpass)#, label=label)
        plt.xlabel('Time [sec]')
        title = '%s, #%d, %.2fs-%.2fs, reff/a99=%.3f ' % (data_name_highK, self.ShotNo, t_st, t_ed, reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1))
        #title_2 = f"\nFitted parameters:y0 = {y0_fit}, A = {A_fit},\nxc = {xc_fit}, w = {w_fit}"
        title_2 = f"\nFitted parameters for Extreme: \ny0 = {y0_fit}±{y0_fit_err}, \nA = {A_fit}±{A_fit_err}, \nxc = {xc_fit}±{xc_fit_err}, \nw = {w_fit}±{w_fit_err}"
        plt.title(title+title_2, fontsize=13, loc='right')
        plt.legend()
        plt.xlim(0, 0.025)
        plt.ylim(0, np.max(final_data)*1.1)
        filename = "highK_condAV_mask_lowpass_t%dto%dms_No%d_fitting.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()
        '''

        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])
        return timedata[:idx_time_period]-timedata[0], reff_a99_highK[idx_time_st:idx_time_ed].mean(axis=-1), y2

    def ana_plot_radhpxi_calThom_condAV_MECH(self, t_st, t_ed, mod_freq, isSaveData=False):
        data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)

        Te = data.getValData('Te')
        #dTe = data.getValData('dTe')
        #ne = data.getValData('n_e')
        #dne = data.getValData('dn_e')
        print('test')

        #plt.rcParams['axes.labelsize'] = 16 #ラベル全体のフォントサイズ
        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')#data_radhinfo.getValData('R')
        ch = data_radhinfo.getValData('pxi')
        rho = data_radhinfo.getValData('rho')
        timedata = data.getDimData('Time')#TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)#-1
        idx_time_period = find_closest(timedata, t_st + 1.0/mod_freq) - idx_time_st
        voltdata_array = np.zeros((len(timedata[:idx_time_period]), 32))
        #fs = 1e3
        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        #freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        freq = fftpack.fftfreq(len(timedata[:idx_time_period]), dt)

        '''
        plt.pcolormesh(timedata, r, Te.T, cmap='jet', vmin=0, vmax=7)
        plt.xlim(4, 4.3)
        cb = plt.colorbar()
        cb.set_label("Te [keV]", fontsize=22)
        cb.ax.tick_params(labelsize=22)
        plt.show()
        '''

        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            voltdata = Te[:,i]#VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            #voltdata_condAV = voltdata[idx_time_st:idx_time_ed].reshape(idx_time_period, -1)
            #voltdata_condAV = voltdata[idx_time_st:idx_time_ed].reshape(np.int(mod_freq*(t_ed-t_st)), -1).T.mean(axis=-1)
            voltdata_condAV = voltdata[idx_time_st:idx_time_ed].reshape(int(mod_freq*(t_ed-t_st)), -1).T.mean(axis=-1)
            #voltdata_condAV = normalize_0_1(voltdata_condAV)
            #voltdata_test = voltdata[idx_time_st:idx_time_st+idx_time_period]
            '''
            low-pass filter for voltdata
            '''
            #yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf = fftpack.rfft(voltdata_condAV) / (int(len(voltdata_condAV) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            #y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata_condAV) / 2)))
            #voltdata_array[:, i] = normalize_0_1(y2)
            voltdata_array[:, i] = y2
            #voltdata_array[:, i] = y2 - y2[:100].mean(axis=-1)
            #voltdata_array[:, i] = voltdata_condAV - voltdata_condAV[:100].mean(axis=-1)
            #plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
            label = 'R=%.3fm' % r[i]
            #plt.plot(timedata[idx_time_st:idx_time_ed], y2, label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_condAV - voltdata_condAV[:100].mean(axis=-1))#, label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_condAV)#, label=label)
            #plt.plot(timedata[:idx_time_period], y2-y2[:100].mean(axis=-1), label=label)
            plt.plot(timedata[:idx_time_period], voltdata_array[:, i], label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_array[:, i], label=label)
        #plt.xlim(4.4, 4.6)
        plt.legend()
        plt.xlabel('Time [sec]', fontsize=20)
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_pxi, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        plt.ylim(0,5)
        filename = "radhpxi_normalized01_t%dto%dms_No%d_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        #plt.savefig(filename)
        plt.show()

        idex_offset = find_closest(timedata, 4.475)

        voltdata_array_deleted = np.delete(voltdata_array, [22,23,25,30,31], 1)
        rho_deleted = np.delete(rho, [22,23,25,30,31], 0)
        R_deleted = np.delete(r, [22,23,25,30,31], 0)
        #indices = find_closest_indices(voltdata_array_deleted[int(idx_time_period/10):int(1*idx_time_period/5.0),:], 0.5)
        #indices = find_closest_indices(voltdata_array_deleted[:int(1*idx_time_period/6),:], 0.5)
        #indices = find_first_greater_indices(voltdata_array_deleted, 0.5)
        indices = find_first_greater_indices(voltdata_array_deleted, 0.03)
        #indices += int(idx_time_period/10)
        #indices = find_closest_indices(voltdata_array_deleted[:int(2*idx_time_period/3),:], 0.5)
        #indices = find_closest_indices(voltdata_array_deleted[:int(2*idx_time_period/8),:], 0.5)
        #print(voltdata_array_deleted[indices, np.arange(len(indices))])
        indice_st = 0
        indice_ed = -5
        popt, pcov = curve_fit(func, 1e3*timedata[indices[indice_st:indice_ed]], rho_deleted[indice_st:indice_ed])
        #popt, pcov = curve_fit(func, 1e3*timedata[indices[1:-1]], np.delete(rho, [22,30], 0)[1:-1])
        perr = np.sqrt((np.diag(pcov)))
        print(popt)
        print(perr)

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dHz), #%d, %.2fs-%.2fs' % (fs, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr')
        #plt.pcolormesh(timedata, r, (voltdata_array - np.mean(voltdata_array[idex_offset:idex_offset+10000,:], axis=0)).T, cmap='bwr', vmin=-0.3, vmax=0.3)
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(timedata[idx_time_st:idx_time_ed-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(1e3*timedata[:idx_time_period], r, voltdata_array.T, cmap='jet')
        #plt.pcolormesh(1e3*timedata[:idx_time_period], rho, voltdata_array.T, cmap='jet')
        #plt.pcolormesh(1e3*timedata[:idx_time_period], np.delete(rho, [22,30], 0), np.delete(voltdata_array, [22,30], 1).T, cmap='jet')
        plt.pcolormesh(1e3*timedata[:idx_time_period]-1e3*timedata[0], rho_deleted, voltdata_array_deleted.T, cmap='jet')
        plt.plot(1e3*timedata[indices]-1e3*timedata[0], rho_deleted, 'b.')
        plt.plot(1e3*timedata[indices[indice_st:indice_ed]]-1e3*timedata[0], rho_deleted[indice_st:indice_ed], 'r.')
        #plt.plot(1e3*timedata[indices], func(1e3*timedata[indices], *popt), 'b')
        plt.plot(1e3*timedata[indices[indice_st:indice_ed]]-1e3*timedata[0], func(1e3*timedata[indices[indice_st:indice_ed]], *popt), 'r')
        #plt.pcolormesh(t, r, Te.T)
        #plt.pcolormesh(t, r[1:-1], (Te[:, 2:]-Te[:, 1:-1]-Te[:, 1:-1]+Te[:, :-2]).T, cmap='bwr')
        cb = plt.colorbar()
        #cb.set_label("d2Te/dr2")
        #cb.set_label(r"$V_{ECE}$ - offset")
        cb.set_label("Te [0, 1]", fontsize=22)
        cb.ax.tick_params(labelsize=22)
        #plt.xlim(0, 1e3*(timedata[idx_time_period]-timedata[0]))
        #plt.ylim(np.min(rho_deleted), np.max(rho_deleted))
        plt.xlabel("Time [ms]", fontsize=22)
        #plt.xlim(4.4, 4.6)
        #plt.ylabel("R [m]", fontsize=22)
        plt.ylabel("reff/a99", fontsize=22)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=22)
        filename = "radhpxi_pcolormesh_normalized01_t%dto%dms_No%d_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        #plt.savefig(filename)
        plt.show()
        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

        dTe = -(np.diff(voltdata_array_deleted))
        dTedreffa99=dTe/np.diff(rho_deleted)
        dTedR=dTe/np.diff(R_deleted)
        # 各y座標に対して、x方向（横軸）の最大値のインデックスを取得
        max_indices = np.argmax(dTe, axis=0)  # 各行ごとの最大値のインデックス

        # 最大値のx座標とy座標を取得
        x_max = 1e3*timedata[:idx_time_period][max_indices]-1e3*timedata[0]

        # 最大値のデータを取得（必要に応じて）
        #data_max = dTe[np.arange(ny), max_indices]

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dHz), #%d, %.2fs-%.2fs\nd(reff/a99)/dt=%.9f+-%.9f' % (fs, self.ShotNo, t_st, t_ed, popt[0], perr[0])
        plt.title(title, fontsize=18, loc='right')
        #plt.pcolormesh(1e3*timedata[:idx_time_period]-1e3*timedata[0], rho_deleted[:-1], -(np.diff(voltdata_array_deleted)).T, cmap='jet', vmin=0, vmax=0.3)
        #plt.pcolormesh(1e3*timedata[:idx_time_period]-1e3*timedata[0], rho_deleted[:-1], dTedreffa99.T, cmap='jet', vmin=0, vmax=10)
        plt.pcolormesh(1e3*timedata[:idx_time_period]-1e3*timedata[0], rho_deleted[:-1], dTedR.T, cmap='jet', vmin=0, vmax=10)
        # 最大値のポイントを赤い丸でプロット
        plt.plot(x_max, rho_deleted[:-1], 'ro', markersize=10, label='Max')
        cb = plt.colorbar()
        cb.set_label("dTe/dR [keV/m]", fontsize=22)
        cb.ax.tick_params(labelsize=22)
        #plt.xlim(0, 1e3*(timedata[idx_time_period]-timedata[0]))
        #plt.ylim(np.min(rho_deleted), np.max(rho_deleted))
        plt.xlabel("Time [ms]", fontsize=22)
        plt.ylabel("reff/a99", fontsize=22)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=22)
        filename = "dTedR_radhpxi_calThom_t%dto%dms_No%d.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.savefig(filename)
        plt.show()

        #offset = np.arange(voltdata_array_deleted.shape[1])
        #plt.plot(1e3*timedata[:idx_time_period], np.diff(voltdata_array_deleted) + 0.1*offset[:-1])
        plt.plot(1e3*timedata[:idx_time_period], -np.diff(voltdata_array_deleted))
        plt.grid(which="both", axis="y")
        plt.ylim(0,0.4)
        plt.show()

        '''
        y = np.delete(rho, [23,30], 0)
        x = 1e3*timedata[:idx_time_period]
        #y = rho
        Z = np.delete(voltdata_array, [23,30], 1)
        #Z = voltdata_array
        #X, Y = np.meshgrid(1e3*timedata[:idx_time_period], rho)
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots()
        #contours = plt.contour(X, Y, voltdata_array.T, 5, colors='black')
        #contours = plt.contour(X, Y, Z.T, 30, colors='black')
        contours = plt.contour(X, Y, Z.T, [0.001, 0.5, 0.999], colors='white')
        contours = plt.contourf(X, Y, Z.T, 30, cmap='jet')
        #im = ax.imshow(voltdata_array.T, extent=[0, 25, np.min(rho), np.max(rho)], origin='lower', cmap='jet', aspect=25)
        #im = ax.pcolormesh(1e3*timedata[:idx_time_period], rho, voltdata_array.T, cmap='jet')
        #im = ax.pcolormesh(x, y, Z.T, cmap='jet')
        #plt.colorbar(im)
        plt.show()
        #self.simulate_heat_propagation(timedata[:idx_time_period], voltdata_array[:,5])
        '''
        if isSaveData:
            filename = "rho_dTeMax_radhpxi_calThom_LP%dHz_t%dto%dms_No%d.txt" % (fs, int(t_st * 1000), int(t_ed * 1000), self.ShotNo)
            np.savetxt(filename, np.c_[rho_deleted[:-1], x_max])

        return 1e3*timedata[:idx_time_period], rho_deleted, np.diff(voltdata_array_deleted)
        #return 1e3*timedata[:idx_time_period], rho_deleted, voltdata_array_deleted

    def ana_plot_radh_condAV_MECH(self, t_st, t_ed, mod_freq):
        #plt.rcParams['axes.labelsize'] = 16 #ラベル全体のフォントサイズ
        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data_radhinfo.getValData('R')
        ch = data_radhinfo.getValData('pxi')
        rho = data_radhinfo.getValData('rho')
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        idx_time_period = find_closest(timedata, t_st + 1.0/mod_freq) - idx_time_st
        voltdata_array = np.zeros((len(timedata[:idx_time_period]), 32))
        #fs = 1e3
        fs = 5e3#5e3
        dt = timedata[1] - timedata[0]
        #freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        freq = fftpack.fftfreq(len(timedata[:idx_time_period]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            #voltdata_condAV = voltdata[idx_time_st:idx_time_ed].reshape(idx_time_period, -1)
            #voltdata_condAV = voltdata[idx_time_st:idx_time_ed].reshape(np.int(mod_freq*(t_ed-t_st)), -1).T.mean(axis=-1)
            voltdata_condAV = voltdata[idx_time_st:idx_time_ed].reshape(int(mod_freq*(t_ed-t_st)), -1).T.mean(axis=-1)
            #voltdata_condAV = normalize_0_1(voltdata_condAV)
            #voltdata_test = voltdata[idx_time_st:idx_time_st+idx_time_period]
            #noise_level, smoothed, residual = estimate_noise_level(timedata[:idx_time_period], normalize_0_1(voltdata_condAV), window_length=31, polyorder=3, plot=False)
            #print(i)
            #print("推定されたノイズレベル（標準偏差）:", noise_level)
            '''
            low-pass filter for voltdata
            '''
            #yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf = fftpack.rfft(voltdata_condAV) / (int(len(voltdata_condAV) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            #y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata_condAV) / 2)))
            voltdata_array[:, i] = normalize_0_1(y2)
            #noise_level, smoothed, residual = estimate_noise_level(timedata[:idx_time_period], normalize_0_1(y2), window_length=31, polyorder=3, plot=True)
            #print("推定されたノイズレベル（標準偏差）(low-pass後）:", noise_level)
            #voltdata_array[:, i] = y2
            #voltdata_array[:, i] = y2 - y2[:100].mean(axis=-1)
            #voltdata_array[:, i] = voltdata_condAV - voltdata_condAV[:100].mean(axis=-1)
            #plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
            label = 'R=%.3fm' % r[i]
            #plt.plot(timedata[idx_time_st:idx_time_ed], y2, label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_condAV - voltdata_condAV[:100].mean(axis=-1))#, label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_condAV)#, label=label)
            #plt.plot(timedata[:idx_time_period], y2-y2[:100].mean(axis=-1), label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_array[:, i] + 0.01*i, label=label)
            #plt.plot(timedata[:idx_time_period], voltdata_array[:, i], label=label)
        #plt.xlim(4.4, 4.6)
        plt.legend()
        plt.xlabel('Time [sec]', fontsize=20)
        title = '%s, #%d, %.2fs-%.2fs' % (data_name_pxi, self.ShotNo, t_st, t_ed)
        plt.title(title, fontsize=18, loc='right')
        filename = "radhpxi_normalized01_t%dto%dms_No%d_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        #plt.savefig(filename)
        plt.show()

        #i_reff = 11
        #noise_level, smoothed, residual = estimate_noise_level(timedata[:idx_time_period], voltdata_array[:, i_reff], window_length=31, polyorder=3, plot=True)
        #print("推定されたノイズレベル（標準偏差）:", noise_level)


        idex_offset = find_closest(timedata, 4.475)

        voltdata_array_deleted = np.delete(voltdata_array, [22,23,25,30,31], 1)
        rho_deleted = np.delete(rho, [22,23,25,30,31], 0)
        #indices = find_closest_indices(voltdata_array_deleted[int(idx_time_period/10):int(1*idx_time_period/5.0),:], 0.5)
        #indices = find_closest_indices(voltdata_array_deleted[:int(1*idx_time_period/6),:], 0.5)
        indices_05 = find_first_greater_indices(voltdata_array_deleted, 0.5)   #Use for half point fitting
        indices = find_first_greater_indices(voltdata_array_deleted, 0.025)   #Use for half point fitting
        #indices = np.argmax(voltdata_array_deleted, axis=0)   #Use for maximum point fitting
        #indices += int(idx_time_period/10)
        #indices = find_closest_indices(voltdata_array_deleted[:int(2*idx_time_period/3),:], 0.5)
        #indices = find_closest_indices(voltdata_array_deleted[:int(2*idx_time_period/8),:], 0.5)
        #print(voltdata_array_deleted[indices, np.arange(len(indices))])
        indice_st = 0
        indice_ed = -8
        indice_ed_05 = -1
        #indice_ed = -5
        popt, pcov = curve_fit(func, 1e3*timedata[indices[indice_st:indice_ed]], rho_deleted[indice_st:indice_ed])
        popt_05, pcov_05 = curve_fit(func, 1e3*timedata[indices_05[indice_st:indice_ed_05]], rho_deleted[indice_st:indice_ed_05])
        #popt, pcov = curve_fit(func, 1e3*timedata[indices[1:-1]], np.delete(rho, [22,30], 0)[1:-1])
        perr = np.sqrt((np.diag(pcov)))
        print(popt)
        print(perr)
        '''
        popt_2, pcov_2 = curve_fit(func, rho_deleted[indice_st:indice_ed], 1e3*timedata[indices[indice_st:indice_ed]])
        #popt, pcov = curve_fit(func, 1e3*timedata[indices[1:-1]], np.delete(rho, [22,30], 0)[1:-1])
        perr_2 = np.sqrt((np.diag(pcov)))
        print(popt_2)
        print(perr_2)
        print(1/popt_2)
        print(1/perr_2)
        '''

        fig = plt.figure(figsize=(8,4), dpi=150)
        title = 'Low-pass(<%dHz), #%d, %.2fs-%.2fs\nd(reff/a99)/dt=%.9f+-%.9f' % (fs, self.ShotNo, t_st, t_ed, popt[0], perr[0])
        plt.title(title, fontsize=18, loc='right')
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr')
        #plt.pcolormesh(timedata, r, (voltdata_array - np.mean(voltdata_array[idex_offset:idex_offset+10000,:], axis=0)).T, cmap='bwr', vmin=-0.3, vmax=0.3)
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(timedata[idx_time_st:idx_time_ed-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(1e3*timedata[:idx_time_period], r, voltdata_array.T, cmap='jet')
        #plt.pcolormesh(1e3*timedata[:idx_time_period], rho, voltdata_array.T, cmap='jet')
        #plt.pcolormesh(1e3*timedata[:idx_time_period], np.delete(rho, [22,30], 0), np.delete(voltdata_array, [22,30], 1).T, cmap='jet')
        plt.pcolormesh(1e3*timedata[:idx_time_period], rho_deleted, voltdata_array_deleted.T, cmap='jet')
        #plt.plot(1e3*timedata[indices], rho_deleted, 'b.')
        plt.plot(1e3*timedata[indices[indice_st:indice_ed]], rho_deleted[indice_st:indice_ed], 'r.', markersize=10)
        plt.plot(1e3*timedata[indices_05[indice_st:indice_ed_05]], rho_deleted[indice_st:indice_ed_05], 'b.', markersize=10)
        #plt.plot(1e3*timedata[indices], func(1e3*timedata[indices], *popt), 'b')
        plt.plot(1e3*timedata[indices[indice_st:indice_ed]], func(1e3*timedata[indices[indice_st:indice_ed]], *popt), 'r', lw=3)
        plt.plot(1e3*timedata[indices_05[indice_st:indice_ed_05]], func(1e3*timedata[indices_05[indice_st:indice_ed_05]], *popt_05), 'b', lw=3)
        #plt.plot(1e3*timedata[indices[indice_st:indice_ed]], func(1e3*timedata[indices[indice_st:indice_ed]], 1/popt_2[0], -popt_2[1]/popt_2[0]), 'blue')
        #plt.pcolormesh(t, r, Te.T)
        #plt.pcolormesh(t, r[1:-1], (Te[:, 2:]-Te[:, 1:-1]-Te[:, 1:-1]+Te[:, :-2]).T, cmap='bwr')
        cb = plt.colorbar()
        #cb.set_label("d2Te/dr2")
        #cb.set_label(r"$V_{ECE}$ - offset")
        cb.set_label("Te [0, 1]", fontsize=22)
        cb.ax.tick_params(labelsize=22)
        plt.xlim(0, 1e3*(timedata[idx_time_period]-timedata[0]))
        #plt.xlim(0, 10)
        #plt.xlim(0, 1e3/mod_freq)
        plt.ylim(np.min(rho_deleted), np.max(rho_deleted))
        plt.xlabel("Time [ms]", fontsize=22)
        #plt.xlim(4.4, 4.6)
        #plt.ylabel("R [m]", fontsize=22)
        plt.ylabel("reff/a99", fontsize=22)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=22)
        filename = "radhpxi_pcolormesh_normalized01_t%dto%dms_No%d_wfitting003_05_v3.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        #filename = "radhpxi_pcolormesh_normalized01_t%dto%dms_No%d_wofitting.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])
        i_reff_1 = 0
        i_reff_2 = 12
        i_reff_3 = 21
        title = 'Low-pass(<%dHz), #%d, %.2fs-%.2fs' % (fs, self.ShotNo, t_st, t_ed)
        plt.title(title)
        label_1 = 'reff/a99=%.2f' % rho_deleted[i_reff_1]
        label_2 = 'reff/a99=%.2f' % rho_deleted[i_reff_2]
        label_3 = 'reff/a99=%.2f' % rho_deleted[i_reff_3]
        plt.plot(1e3*(timedata[:idx_time_period]-timedata[0]), voltdata_array_deleted[:, i_reff_1], label=label_1)
        plt.plot(1e3*(timedata[:idx_time_period]-timedata[0]), voltdata_array_deleted[:, i_reff_2], label=label_2)
        plt.plot(1e3*(timedata[:idx_time_period]-timedata[0]), voltdata_array_deleted[:, i_reff_3], label=label_3)
        plt.legend()
        plt.show()
        combined_voltdata_array_deleted = np.column_stack((1e3*(timedata[:idx_time_period]-timedata[0]),voltdata_array_deleted[:, i_reff_1],voltdata_array_deleted[:, i_reff_2],voltdata_array_deleted[:, i_reff_3],))
        file_title = 'time_normalized_Te_ra%.2f_ra%.2f_ra%.2f_sn%d.txt' % (rho_deleted[i_reff_1],rho_deleted[i_reff_2],rho_deleted[i_reff_3],self.ShotNo)
        np.savetxt(file_title,combined_voltdata_array_deleted)

        plt.pcolormesh(1e3*timedata[:idx_time_period], rho_deleted[:-1], (np.diff(voltdata_array_deleted)).T, cmap='bwr', vmin=-0.1, vmax=0.1)
        cb = plt.colorbar()
        cb.set_label("dTe [0, 1]", fontsize=22)
        cb.ax.tick_params(labelsize=22)
        plt.xlim(0, 1e3*(timedata[idx_time_period]-timedata[0]))
        plt.ylim(np.min(rho_deleted), np.max(rho_deleted))
        plt.xlabel("Time [ms]", fontsize=22)
        plt.ylabel("reff/a99", fontsize=22)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=22)
        plt.show()

        #offset = np.arange(voltdata_array_deleted.shape[1])
        #plt.plot(1e3*timedata[:idx_time_period], np.diff(voltdata_array_deleted) + 0.1*offset[:-1])
        plt.plot(1e3*timedata[:idx_time_period], np.diff(voltdata_array_deleted))
        plt.grid(which="both", axis="y")
        plt.show()

        '''
        y = np.delete(rho, [23,30], 0)
        x = 1e3*timedata[:idx_time_period]
        #y = rho
        Z = np.delete(voltdata_array, [23,30], 1)
        #Z = voltdata_array
        #X, Y = np.meshgrid(1e3*timedata[:idx_time_period], rho)
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots()
        #contours = plt.contour(X, Y, voltdata_array.T, 5, colors='black')
        #contours = plt.contour(X, Y, Z.T, 30, colors='black')
        contours = plt.contour(X, Y, Z.T, [0.001, 0.5, 0.999], colors='white')
        contours = plt.contourf(X, Y, Z.T, 30, cmap='jet')
        #im = ax.imshow(voltdata_array.T, extent=[0, 25, np.min(rho), np.max(rho)], origin='lower', cmap='jet', aspect=25)
        #im = ax.pcolormesh(1e3*timedata[:idx_time_period], rho, voltdata_array.T, cmap='jet')
        #im = ax.pcolormesh(x, y, Z.T, cmap='jet')
        #plt.colorbar(im)
        plt.show()
        #self.simulate_heat_propagation(timedata[:idx_time_period], voltdata_array[:,5])
        '''

        #return 1e3*timedata[:idx_time_period], rho_deleted, np.diff(voltdata_array_deleted)
        return 1e3*timedata[:idx_time_period], rho_deleted, voltdata_array_deleted

    def ana_plot_pci_condAV_MECH(self, t_st, t_ed, mod_freq):
        data = AnaData.retrieve('ece_fast', self.ShotNo, 1)
        r = data.getDimData('R')
        vnum = data.getValNo()
        t_ece = data.getDimData('Time')
        Te = data.getValData('Te')

        data = AnaData.retrieve('pci_profs', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('R')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')
        t = data.getDimData('time')
        reff_a99 = data.getDimData('reff/a99')
        ntilde_pci = data.getValData(0)
        v_pci = data.getValData(1)
        #cosv_pci = data.getValData(2)
        #fig = plt.figure(figsize=(12,6), dpi=150)
        #plt.pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet')
        #plt.xlim(4.0, 4.65)
        #plt.show()

        data_pci_sdens_rvt = AnaData.retrieve('pci_sdens_rvt', self.ShotNo, 1)
        data_pci_sdens_rkt = AnaData.retrieve('pci_sdens_rkt', self.ShotNo, 1)
        t_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('time')
        t_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('time')
        v_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('v')
        k_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('k')
        reff_a99_pci_sdens_rvt = data_pci_sdens_rvt.getDimData('reff/a99')
        reff_a99_pci_sdens_rkt = data_pci_sdens_rkt.getDimData('reff/a99')
        sdens_rvt = data_pci_sdens_rvt.getValData(0)
        sdens_rkt = data_pci_sdens_rkt.getValData(0)
        #i_time=1380
        #images = []

        idx_time_st = find_closest(t, t_st)
        idx_time_ed = find_closest(t, t_ed)
        idx_time_period = find_closest(t, t_st + 1/mod_freq) - idx_time_st
        i_wave = np.int(Decimal(str(mod_freq))*(Decimal(str(t_ed))-Decimal(str(t_st))))
        ntilde_pci_condAV = ntilde_pci[idx_time_st:idx_time_ed].reshape(idx_time_period, i_wave, len(reff_a99)).T.mean(axis=1)
        ntilde_pci_condAV_normalized = np.array(normalize_all_columns_np(ntilde_pci_condAV.T)).T
        v_pci_condAV = v_pci[idx_time_st:idx_time_ed].reshape(idx_time_period, i_wave, len(reff_a99)).T.mean(axis=1)

        '''
        title = 'PCI(ntilde), #%d, %.2fs-%.2fs' % (self.ShotNo, t_st, t_ed)
        #plt.pcolormesh(t[:idx_time_period] - t[0], reff_a99, ntilde_pci_condAV, cmap='jet')
        plt.pcolormesh(t[:idx_time_period] - t[0], reff_a99, ntilde_pci_condAV_normalized, cmap='jet')
        plt.xlabel('Time [s]')
        plt.ylabel('reff/a99')
        plt.ylim(-1.2, 1.2)
        plt.title(title)
        #plt.show()
        filename = 'pci_condAV_normalized_%d.png' % (self.ShotNo)
        plt.savefig(filename)
        '''

        '''
        i_pci=133
        label = 'reff/a99=%.2f' % reff_a99[i_pci]
        #plt.plot(t[:idx_time_period] - t[0],  ntilde_pci_condAV[i_pci], label=label)
        plt.plot(t[:idx_time_period] - t[0],  ntilde_pci_condAV_normalized[i_pci], label=label)
        i_pci=143
        label = 'reff/a99=%.2f' % reff_a99[i_pci]
        #plt.plot(t[:idx_time_period] - t[0],  ntilde_pci_condAV[i_pci], label=label)
        plt.plot(t[:idx_time_period] - t[0],  ntilde_pci_condAV_normalized[i_pci], label=label)
        i_pci=153
        label = 'reff/a99=%.2f' % reff_a99[i_pci]
        #plt.plot(t[:idx_time_period] - t[0],  ntilde_pci_condAV[i_pci], label=label)
        plt.plot(t[:idx_time_period] - t[0],  ntilde_pci_condAV_normalized[i_pci], label=label)
        plt.xlabel('Time [s]')
        plt.ylabel('ntilde')
        plt.title(title)
        plt.legend()
        plt.show()
        '''
        '''
        plt.figure(figsize=(10, 6), dpi=150)
        plt.title(title)
        plt.plot(reff_a99, ntilde_pci_condAV + 50000*np.arange(25))
        plt.grid(axis='x', which='both')
        plt.xlabel('reff/a99')
        plt.ylabel('ntilde_pci(Bottom:0s, Top:25ms)')
        filename = 'pci_condAV_%d.png' % (self.ShotNo)
        #plt.savefig(filename)
        plt.show()
        '''

        sdens_rkt_condAV = sdens_rkt[idx_time_st:idx_time_ed].reshape(idx_time_period, i_wave, len(reff_a99_pci_sdens_rkt), len(k_pci_sdens_rkt)).mean(axis=1)
        sdens_rvt_condAV = sdens_rvt[idx_time_st:idx_time_ed].reshape(idx_time_period, i_wave, len(reff_a99_pci_sdens_rvt), len(v_pci_sdens_rvt)).mean(axis=1)

        #plt.imshow(sdens_rkt_condAV[10], cmap='jet')
        '''
        plt.pcolormesh( reff_a99, k_pci_sdens_rkt, sdens_rkt_condAV[10].T, cmap='jet')
        plt.show()
        plt.pcolormesh( reff_a99, v_pci_sdens_rvt, sdens_rvt_condAV[10].T, cmap='jet')
        plt.show()
        '''
        for i_time in range(idx_time_period):
            fig, axarr = plt.subplots(5, 1, gridspec_kw={'wspace': 0, 'hspace': 0.25, 'height_ratios': [4,4,1,1,1]}, figsize=(8, 11), dpi=150,
                                      sharex=False)
            # pp.set_clim(1e1, 1e3)
            axarr[1].set_xlabel('reff/a99', fontsize=10)
            axarr[0].set_ylabel('k [1/mm]', fontsize=10)
            axarr[1].set_ylabel('v [km/s]', fontsize=10)
            axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[0].set_title('#%d, t=%.3fsec' % (self.ShotNo, t_pci_sdens_rkt[i_time]-t_pci_sdens_rkt[0]), fontsize=20, loc='left')
            mappable0 = axarr[0].pcolormesh(reff_a99_pci_sdens_rkt, k_pci_sdens_rkt, sdens_rkt_condAV[i_time].T, cmap='jet', norm=LogNorm(vmin=1e3, vmax=1e5))
            mappable1 = axarr[1].pcolormesh(reff_a99_pci_sdens_rvt, v_pci_sdens_rvt, sdens_rvt_condAV[i_time].T, cmap='jet', norm=LogNorm(vmin=1e-2, vmax=1e0))
            pp0 = fig.colorbar(mappable0, ax=axarr[0], orientation="vertical", pad=0.01)
            pp1 = fig.colorbar(mappable1, ax=axarr[1], orientation="vertical", pad=0.01)
            pp0.set_label('PSD', fontname="Arial", fontsize=10)
            pp1.set_label('PSD', fontname="Arial", fontsize=10)
            #plt.show()

            axarr[2].set_ylabel('reff/a99', fontsize=10)
            axarr[3].set_ylabel('reff/a99', fontsize=10)
            axarr[4].set_ylabel('Te [keV]', fontsize=10)
            axarr[4].set_xlabel('Time [sec]', fontsize=10)
            axarr[3].set_xlim(0.0, 0.0250)
            axarr[2].set_xlim(0.0, 0.0250)
            axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            im0 = axarr[2].pcolormesh(t[:idx_time_period]-t[0], reff_a99, ntilde_pci_condAV, cmap='jet',
                                      norm=LogNorm(vmin=1e5, vmax=1e6))
            im1 = axarr[3].pcolormesh(t[:idx_time_period]-t[0], reff_a99, v_pci_condAV, cmap='bwr', vmin=-2, vmax=2)
            pp0 = fig.colorbar(im0, ax=axarr[2], pad=0.01)
            pp1 = fig.colorbar(im1, ax=axarr[3], pad=0.01)
            pp0.set_label('ntilde[a.u.]', fontname="Arial", fontsize=10)
            pp1.set_label('v [km/sec]', fontname="Arial", fontsize=10)

            arr_erace = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 32, 33, 44,
                         45, 46, 47, 50, 51, 58]
            arr_erace = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
            axarr[4].plot(t_ece, Te[:, arr_erace])
            axarr[4].set_ylim(0, 5)
            axarr[4].set_xlim(t_st, t_st+0.03)
            axarr[4].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[2].axvline(t_pci_sdens_rkt[i_time]-t_pci_sdens_rkt[0], color='red')
            axarr[3].axvline(t_pci_sdens_rkt[i_time]-t_pci_sdens_rvt[0], color='red')
            axarr[4].axvline(t_pci_sdens_rkt[i_time]-t_pci_sdens_rkt[0]+t_st, color='red')
            # plt.savefig("fmp_psd_No%d.png" % self.ShotNo)
            filename = 'pci_condAV_k_v_reff_time_%d_t%.4f.png' % (self.ShotNo, t_pci_sdens_rkt[i_time]-t_pci_sdens_rkt[0])
            plt.savefig(filename)
            plt.close()


    def normalize_highK_w_gradTe(self, t_st, t_ed, mod_freq, isSaveData=False):
        timedata_highK, rho_highK, data_highK = self.ana_plot_highK_condAV_MECH(t_st, t_ed, mod_freq)
        #timedata_gradTe, rho_gradTe, data_gradTe = self.ana_plot_radh_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_gradTe, rho_gradTe, data_gradTe = self.ana_plot_radhpxi_calThom_condAV_MECH(t_st, t_ed, mod_freq)
        timedata_highK *= 1e3
        timedata_highK_ = np.linspace(0, 1/mod_freq, len(timedata_highK))
        timedata_gradTe_ = np.linspace(0, 1/mod_freq, len(timedata_gradTe))
        f = interp1d(timedata_highK_, data_highK, kind='cubic')
        data_highK_extended = f(timedata_gradTe_)
        #plt.plot(timedata_highK_, data_highK)
        #plt.plot(timedata_gradTe_, data_highK_extended+0.001)
        print(rho_gradTe)
        i_rho =20#13#19#17#15#9#12#16#7#20
        # 1. Aの0.003-0.005秒の間の平均値をA全体から差し引く
        # 0.003sから0.005sのデータを除外
        exclude_start_time = 0.0031
        exclude_end_time = 0.0049
        idx_A_range = np.where((timedata_gradTe_>= exclude_start_time) & (timedata_gradTe_<= exclude_end_time))
        mean_A_range = np.mean(data_highK_extended[idx_A_range])
        A2 = data_highK_extended- mean_A_range

        # 2. Bが最大値を取る時刻の前後0.001秒のAの値の平均値を求める
        max_B_time =timedata_gradTe_[np.argmax(-1*data_gradTe[:, i_rho])]
        idx_A_near_max_B = np.where((timedata_gradTe_>= (max_B_time - 0.003)) & (timedata_gradTe_<= (max_B_time + 0.003)))
        mean_A_near_max_B = np.mean(A2[idx_A_near_max_B])

        # 3. Bを0からmean_A_near_max_Bの範囲に規格化
        B2 = (-1*data_gradTe[:, i_rho] - np.min(-1*data_gradTe[:, i_rho])) / (np.max(-1*data_gradTe[:, i_rho]) - np.min(-1*data_gradTe[:, i_rho])) * mean_A_near_max_B
        B2 = -1*data_gradTe[:, i_rho] / np.max(-1*data_gradTe[:, i_rho]) * mean_A_near_max_B

        # 4. A2からB2を差し引いたデータCを求める
        C = A2 - B2


        label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho]
        plt.plot(timedata_gradTe_, data_gradTe[:, i_rho], label='gradTe(' + label_gradTe)
        label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho+1]
        plt.plot(timedata_gradTe_, data_gradTe[:, i_rho+1], label='gradTe(' + label_gradTe)
        label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho+2]
        plt.plot(timedata_gradTe_, data_gradTe[:, i_rho+2], label='gradTe(' + label_gradTe)
        plt.xlabel('Time [msec]')
        plt.legend()
        plt.show()

        #label_highK = 'r/a=%.3f)(3-30kHz)' % rho_highK
        label_highK = 'r/a=%.3f)(20-200kHz)' % rho_highK
        label_gradTe = 'r/a=%.3f)' % ((rho_gradTe[i_rho]+rho_gradTe[i_rho+1])/2)
        '''
        title = '#%d, %.2fs-%.2fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        #label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho]
        plt.plot(timedata_gradTe_, normalize_0_1(-1*data_gradTe[:, i_rho]), label='gradTe(' + label_gradTe, color='blue')
        #label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho+1]
        #plt.plot(timedata_gradTe_, normalize_0_1(-1*data_gradTe[:, i_rho+1]), label='gradTe(' + label_gradTe, color='green')
        plt.plot(timedata_gradTe_, normalize_0_1(data_highK_extended), label='highK(' + label_highK, color='red')
        plt.xlabel('Time [msec]')
        plt.legend()
        plt.show()
        '''


        title = '#%d, %.2fs-%.2fs' % (self.ShotNo, t_st, t_ed)
        fig = plt.figure()
        plt.title(title)
        ax1 = fig.add_subplot(111)
        #label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho]
        ln1=ax1.plot(timedata_gradTe_, -1*data_gradTe[:, i_rho], label='gradTe(' + label_gradTe, color='blue')
        ax2 = ax1.twinx()
        #label_gradTe = 'r/a=%.3f)' % rho_gradTe[i_rho+1]
        #plt.plot(timedata_gradTe_, normalize_0_1(-1*data_gradTe[:, i_rho+1]), label='gradTe(' + label_gradTe, color='green')
        ln2=ax2.plot(timedata_gradTe_, data_highK_extended, label='highK(' + label_highK, color='red')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='lower right')
        ax1.set_ylabel('grad T_e [keV/m]')
        ax1.grid(True)
        ax2.set_ylabel('e-scale turb. [a.u.]')
        ax1.set_xlabel('Time [msec]')
        ax1.set_ylim(0,0.13)
        #plt.legend()
        plt.show()

        fig = plt.figure(figsize=(6,4.5), dpi=150)
        title = '#%d, %.2fs-%.2fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        #plt.plot(timedata_gradTe_, 1*data_gradTe[:, i_rho]+data_highK_extended*0.12/0.0007)
        #plt.plot(timedata_gradTe_, 1*data_gradTe[:, i_rho]*0.0007/0.12+data_highK_extended)
        plt.plot(timedata_gradTe_, A2, 'r-', label='e-scale turb.(' + label_highK)
        plt.plot(timedata_gradTe_[idx_A_range], A2[idx_A_range], color='black')
        plt.plot(timedata_gradTe_, C, 'g-', label='coupling comp. of turb.')
        plt.plot(timedata_gradTe_, B2, 'b-', label='normalized gradTe('+label_gradTe)
        plt.grid(True)
        plt.legend()
        plt.xlabel("Time [s]", fontsize=18)
        filename = 'turb_gradTe_coupling_condAV_%d_lowfreq.png' % self.ShotNo
        plt.savefig(filename)
        plt.xlim(0, np.max(timedata_gradTe_))
        plt.show()

        '''
        fig = plt.figure(figsize=(6,4.5), dpi=150)
        title = '#%d, %.2fs-%.2fs' % (self.ShotNo, t_st, t_ed)
        plt.title(title)
        #plt.plot(timedata_gradTe_, 1*data_gradTe[:, i_rho]+data_highK_extended*0.12/0.0007)
        #plt.plot(timedata_gradTe_, 1*data_gradTe[:, i_rho]*0.0007/0.12+data_highK_extended)
        plt.plot(timedata_gradTe_, A2, 'r-', label='e-scale turb.(' + label_highK)
        plt.plot(timedata_gradTe_[idx_A_range], A2[idx_A_range], color='black')
        plt.plot(timedata_gradTe_, C, 'g-', label='coupling comp. of turb.')
        plt.plot(timedata_gradTe_, B2, 'b-', label='normalized gradTe('+label_gradTe)
        plt.grid(True)
        plt.legend()
        plt.xlabel("Time [s]", fontsize=18)
        plt.xlim(0, 0.1)
        filename = 'turb_gradTe_coupling_condAV_expand_%d.png' % self.ShotNo
        plt.savefig(filename)
        plt.show()
        '''

        '''
        plt.plot(timedata_gradTe_, -1*data_highK_extended/data_gradTe[:, i_rho])
        plt.ylim(-1,1)
        plt.show()
        plt.title(title + ', gradTe(' + label_gradTe + ', turb.(' + label_highK)
        plt.plot(-1*data_gradTe[:, i_rho], data_highK_extended, color='red')
        plt.xlabel('gradTe')
        plt.ylabel('e-scale turb.')
        plt.show()
        '''
        if isSaveData:
            #t_data_gradTe = np.stack((timedata_gradTe_, normalize_0_1(-1*data_gradTe[:, i_rho])))
            t_data_gradTe = np.stack((timedata_gradTe_, -1*data_gradTe[:, i_rho]))
            t_data_highK = np.stack((timedata_gradTe_, normalize_0_1(data_highK_extended)))
            data_gradTe_highK = np.stack((-1*data_gradTe[:, i_rho], data_highK_extended))
            filename = '_%d.txt' % (self.ShotNo)
            filename_gradTe = '_ra%.3f' %  rho_gradTe[i_rho]
            filename_highK = '_ra%.3f' % rho_highK
            np.savetxt('time_gradTe' + filename_gradTe + filename, t_data_gradTe.T)
            np.savetxt('time_highK' + filename_highK + filename, t_data_highK.T)
            np.savetxt('gradTe_highK' + filename_gradTe + filename_highK + filename, data_gradTe_highK.T)
            np.savetxt('gradTe_2D' + filename, -1*data_gradTe)

    def simulate_heat_propagation(self, t, y):
        y_2D = np.zeros((2*len(y), 20))
        t_2 = np.linspace(0, np.max(t)*2, 2*len(t))
        for i in range(20):
            y_2D[40*i:40*i+len(y), i] = y
            #y_2D[750 + 60*i:750 + 60*i+len(y), i] += y
            #y_2D[1500 + 80*i:1500 + 80*i+len(y), i] += y
            #y_2D[2250 + 100*i:2250 + 100*i+len(y), i] += y
            #y_2D[3000 + 120*i:3000 + 120*i+len(y), i] += y
            #y_2D[3750 + 140*i:3750 + 140*i+len(y), i] += y
            #y_2D[4500 + 160*i:4500 + 160*i+len(y), i] += y
            #y_2D[:, i] = normalize_0_1(y_2D[:,i])
            #y_2D[160*i:160*i+len(y), i] = y
            #y_2D[750 +160*i:750 +160*i+len(y), i] += y
            #y_2D[1500 +160*i:1500 +160*i+len(y), i] += y
            #y_2D[2250 +160*i:2250 +160*i+len(y), i] += y
            #y_2D[3000 +160*i:3000 +160*i+len(y), i] += y
            #y_2D[3750 +160*i:3750 +160*i+len(y), i] += y
            #y_2D[4500 +160*i:4500 +160*i+len(y), i] += y
            #y_2D[:, i] = normalize_0_1(y_2D[:,i])
        rho = np.linspace(0, 1, 20)

        fig = plt.figure(figsize=(8,6), dpi=150)
        #title = 'Low-pass(<%dHz), #%d, %.2fs-%.2fs' % (fs, self.ShotNo, t_st, t_ed)
        #plt.title(title, fontsize=18, loc='right')
        print(t_2)
        print(rho)
        print(y_2D.shape)
        plt.pcolormesh(1e3*t_2, rho, y_2D.T, cmap='jet')
        cb = plt.colorbar()
        cb.set_label("Te [0, 1]", fontsize=22)
        cb.ax.tick_params(labelsize=22)
        #plt.xlim(0, 1e3*(timedata[idx_time_period]-timedata[0]))
        plt.xlabel("Time [ms]", fontsize=22)
        plt.ylabel("reff/a99", fontsize=22)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=22)
        #filename = "radhpxi_pcolormesh_normalized01_t%dto%dms_No%d_v2.png" %(int(t_st*1000), int(t_ed*1000), self.ShotNo)
        #plt.savefig(filename)
        plt.show()

    def ana_plot_ece_qe(self, isSaveData=False):
        data = AnaData.retrieve('ece_fast_qe', self.ShotNo, 1)

        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))

        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        num_val = 5
        print(num_val, data.getValName(num_val), data.getValUnit(num_val))
        data_ece_fast_qe = data.getValData(num_val)
        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('time')
        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        reff = data.getDimData('reff')
        Te = data.getValData('Te')
        ne = data.getValData('ne')
        Q_ech_ov_n = data.getValData('Q_ECH/n')
        Qetot_ov_n = data.getValData('Qetot/n')
        #r_unit = data.getDimUnit('m')

        plt.plot(t, ne)
        plt.ylim(0, 2)
        plt.show()
        #plt.pcolormesh(t, reff, Qetot_ov_n.T, cmap='jet', vmin=0, vmax=50)
        plt.pcolormesh(t, reff, data_ece_fast_qe.T, cmap='jet', vmin=0, vmax=100)
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.xlabel('Time [sec]')
        plt.ylabel('reff [m]')
        cbar = plt.colorbar(pad=0.01)
        cbar.set_label(data.getValName(num_val) + " [" + data.getValUnit(num_val) + "]", fontsize=18, rotation=270, labelpad=16)
        cbar.ax.get_yaxis().labelpad = 15
        plt.xlim(4.0, 4.1)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        plt.show()

    def condAV_qe_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None,
                                              prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188796
        for i in range(shot_st, shot_ed + 1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh, t_qe, reff_qe, buf_cond_av_qe, val_name_qe, val_unit_qe, rho_radh, num_av, fs = self.condAV_qe_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh_ac = buf_cond_av_radh - np.mean(buf_cond_av_radh[-1000:, :], axis=0)
                    buf_cond_av_radh *= num_av
                    buf_cond_av_radh_ac *= num_av
                    buf_cond_av_qe *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_, t_qe,reff_qe,  buf_cond_av_qe_, val_name_qe, val_unit_qe, rho_radh, num_av, fs = self.condAV_qe_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_radh_ -= np.mean(buf_cond_av_radh_[-1000:, :], axis=0)
                    buf_cond_av_radh_ac += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_qe += (buf_cond_av_qe_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh = buf_cond_av_radh / buf_num_av
        cond_av_radh_ac = buf_cond_av_radh_ac / buf_num_av
        cond_av_qe = buf_cond_av_qe / buf_num_av

        i_ch = 16
        # label_highK = 'highK(reff/a99=%.3f)' % (reff_a99_highK.mean(axis=-1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        b = ax2.plot( t_qe, cond_av_qe[:, 22], 'g-', label=val_name_qe, zorder=2)
        #b = ax2.plot( buf_t_qe, cond_av_qe[:, 18], '-', label=data_qe.getValName(num_val)+"18", zorder=2)
        #b = ax2.plot( buf_t_qe, cond_av_qe[:, 20], '-', label=data_qe.getValName(num_val)+"20", zorder=2)
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        plt.title('Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel(val_name_qe + ' [' + val_unit_qe + "]", rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        # ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        # plt.savefig("timetrace_condAV_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        # plt.savefig("timetrace_condAV_ece_radh_highK_mp_No%dto%d_prom%.1fto%.1f_v3.png" % (shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()


        fig = plt.figure(figsize=(8,6), dpi=150)
        plt.title('Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        plt.pcolormesh(1e3*t_qe, reff_qe, cond_av_qe.T, cmap='bwr', vmin=-50, vmax=50)
        cb = plt.colorbar()
        cb.set_label(val_name_qe + ' [' + val_unit_qe + "]", fontsize=18, rotation=270, labelpad=16)
        #cb.set_label("Te [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("condAV_radh_ece_LP%dkHz_No%dto%d_prom%.1fto%.1f_v4.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()

    def condAV_qe_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        t = data.getDimData('Time')

        data_qe = AnaData.retrieve('ece_fast_qe', self.ShotNo, 1)

        dnum = data_qe.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data_qe.getValNo()
        for i in range(vnum):
            print(i, data_qe.getValName(i), data_qe.getValUnit(i))
        t_qe = data_qe.getDimData('time')
        reff_qe = data_qe.getDimData('reff')
        num_val =5
        print(num_val, data_qe.getValName(num_val), data_qe.getValUnit(num_val))
        data_ece_fast_qe = data_qe.getValData(num_val)


        #r = data_radhinfo.getValData('rho')
        #r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))


        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                                   timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_qe = t_qe[find_closest( t_qe, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_qe, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

                buf_qe = data_ece_fast_qe[find_closest( t_qe, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_qe, timedata_peaks_ltd[i]+t_ed_condAV), :]

            else:
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_qe = t_qe[find_closest( t_qe, timedata_peaks_ltd[i] - t_st_condAV):find_closest( t_qe,
                                                                                                                      timedata_peaks_ltd[
                                                                                                                          i] + t_ed_condAV)] - \
                                timedata_peaks_ltd[i]
                    buf_qe += data_ece_fast_qe[find_closest( t_qe, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_qe, timedata_peaks_ltd[i]+t_ed_condAV),:]
                except Exception as e:
                    print(e)

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_qe = buf_qe/len(timedata_peaks_ltd)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        a = ax2.plot( buf_t_qe, cond_av_qe[:, 22], 'g-', label=data_qe.getValName(num_val)+"22", zorder=2)
        #a = ax2.plot( buf_t_qe, cond_av_qe[:, 18], '-', label=data_qe.getValName(num_val)+"18", zorder=2)
        #a = ax2.plot( buf_t_qe, cond_av_qe[:, 20], '-', label=data_qe.getValName(num_val)+"20", zorder=2)
        plt.title('Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel(data_qe.getValName(num_val) + " [" + data_qe.getValUnit(num_val) + "]", rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Prominence:%.1fto%.1f' % (prom_min, prom_max)
        plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*buf_t_qe, reff_qe, cond_av_qe.T, cmap='jet', vmin=0, vmax=5)
        cb = plt.colorbar()
        cb.set_label(data_qe.getValName(num_val) + " [" + data_qe.getValUnit(num_val) + "]", fontsize=18, rotation=270, labelpad=16)
        #cb.set_label("Te [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("condAV_radh_ece_LP%dkHz_No%dto%d_prom%.1fto%.1f_v4.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()



        return buf_t_radh, cond_av_radh, buf_t_qe, reff_qe, cond_av_qe,data_qe.getValName(num_val), data_qe.getValUnit(num_val), rho, len(timedata_peaks_ltd), fs

    def condAV_nel_trigLowPassRadh_multishots(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None,
                                              prom_min=None, prom_max=None):
        j = 0
        shot_st = 188780
        shot_ed = 188796
        for i in range(shot_st, shot_ed + 1):
            try:
                self.ShotNo = i
                print(self.ShotNo)
                if j == 0:
                    t_radh, buf_cond_av_radh, t_nel, R_fir_mm, buf_cond_av_nel, val_name_nel, val_unit_nel, rho_radh, num_av, fs = self.condAV_nel_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh_ac = buf_cond_av_radh - np.mean(buf_cond_av_radh[-1000:, :], axis=0)
                    buf_cond_av_radh *= num_av
                    buf_cond_av_radh_ac *= num_av
                    buf_cond_av_nel *= num_av
                    buf_num_av = num_av
                    print(buf_num_av)
                else:
                    t_radh, buf_cond_av_radh_, t_nel, R_fir_mm, buf_cond_av_nel_, val_name_nel, val_unit_nel, rho_radh, num_av, fs = self.condAV_nel_trigLowPassRadh(
                        t_st=t_st,
                        t_ed=t_ed,
                        prom_min=0.4,
                        prom_max=1.0, Mode_ece_radhpxi_calThom=True)
                    buf_cond_av_radh += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_radh_ -= np.mean(buf_cond_av_radh_[-1000:, :], axis=0)
                    buf_cond_av_radh_ac += (buf_cond_av_radh_ * num_av)
                    buf_cond_av_nel += (buf_cond_av_nel_ * num_av)
                    buf_num_av += num_av
                    print(num_av)
                    print(buf_num_av)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("%s, %s, %s" % (exc_type, fname, exc_tb.tb_lineno))
                print(e)
            j += 1
            print(j)

        cond_av_radh = buf_cond_av_radh / buf_num_av
        cond_av_radh_ac = buf_cond_av_radh_ac / buf_num_av
        cond_av_nel = buf_cond_av_nel / buf_num_av
        #cond_av_nel = np.array(normalize_all_columns_np(cond_av_nel_))


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        i_ch = 16
        label_radh = "ECE(rho=" + str(rho_radh[i_ch]) + ")"
        c = ax1.plot(t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        #a = ax2.plot( buf_t_nel, cond_av_nel[:], 'g-', label=data_nel.getValName(num_val), zorder=2)
        for i in range(2, len(cond_av_nel[0,:])):
            a = ax2.plot( t_nel, cond_av_nel[:,i]- np.mean(cond_av_nel[:200,i]) + 0.01*i, '-', zorder=2, label=R_fir_mm[i-2])
            #a = ax2.plot( t_nel, cond_av_nel[:,i]- np.mean(cond_av_nel[:200,i]), '-', zorder=2, label=R_fir_mm[i-2])
        #a = ax2.plot( buf_t_qe, cond_av_qe[:, 18], '-', label=data_qe.getValName(num_val)+"18", zorder=2)
        #a = ax2.plot( buf_t_qe, cond_av_qe[:, 20], '-', label=data_qe.getValName(num_val)+"20", zorder=2)
        plt.title('Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel("nel [" + val_unit_nel + "]", rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(8,6), dpi=150)
        plt.title('Prominence:%.1fto%.1f, #%d-%d' % (prom_min, prom_max, shot_st, shot_ed), loc='right')
        test = (cond_av_nel[:, 2:]- np.mean(cond_av_nel[200:,2:])).T
        plt.pcolormesh(1e3*t_nel, R_fir_mm, (cond_av_nel[:, 2:]- np.mean(cond_av_nel[:200,2:], axis=0)).T, cmap='bwr', vmin=-0.008, vmax=0.008)
        cb = plt.colorbar()
        cb.set_label("nel [e19m-2]", fontsize=18, rotation=270, labelpad=16)
        #cb.set_label("Te [0,1]", fontsize=18)
        #cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel("R [mm]", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        #plt.savefig("condAV_radh_LP%dkHz_No%dto%d_prom%.1fto%.1f_v3.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        #plt.savefig("condAV_radh_ece_LP%dkHz_No%dto%d_prom%.1fto%.1f_v4.png" % (fs/1e3,shot_st, shot_ed, prom_min, prom_max))
        plt.show()
        plt.close()



    def condAV_nel_trigLowPassRadh(self, t_st=None, t_ed=None, arr_peak_selection=None, arr_toff=None, prom_min=None, prom_max=None, Mode_ece_radhpxi_calThom=None):
        if Mode_ece_radhpxi_calThom:
            data = AnaData.retrieve('ece_radhpxi_calThom', self.ShotNo, 1)
            Te = data.getValData('Te')
            r = data.getDimData('R')
            timedata = data.getDimData('Time')  # TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        else:
            data_name_pxi = 'RADHPXI'  # RADLPXI
            data_name_info = 'radhinfo'  # radlinfo
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()

        data_name_info = 'radhinfo'  # radlinfo
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        t = data.getDimData('Time')

        data_nel = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        dnum = data_nel.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data_nel.getValNo()
        t_nel = data_nel.getDimData('Time')
        data_fir_nel = np.zeros((len(t_nel), vnum))
        R_fir_mm = np.zeros(vnum-2)
        for i in range(vnum):
            print(i, data_nel.getValName(i), data_nel.getValUnit(i))
            data_fir_nel[:,i] = data_nel.getValData(i)
            if i > 1:
                R_fir_mm[i-2] = int(data_nel.getValName(i)[3:-1])
        num_val =4
        print(num_val, data_nel.getValName(num_val), data_nel.getValUnit(num_val))


        #r = data_radhinfo.getValData('rho')
        #r = data_radhinfo.getValData('R')
        rho = data_radhinfo.getValData('rho')
        ch = data_radhinfo.getValData('pxi')

        i_ch = 16#14#13#9
        #voltdata_array = np.zeros((len(timedata), 32))
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        voltdata_array_calThom = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))


        fs = 5e3#2e3#5e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        #fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            #voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            if Mode_ece_radhpxi_calThom:
                voltdata = Te[:,i]
            else:
                voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, int(ch[i])).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = normalize_0_1(y2)
            if Mode_ece_radhpxi_calThom:
                voltdata_array_calThom[:, i] = y2


        #_, _, _, timedata_peaks_ltd_,_,_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
        if Mode_ece_radhpxi_calThom:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=True, Mode_find_peaks='avalanche20240403_ece', prom_min=prom_min, prom_max=prom_max)
        else:
            peaks, prominences, timedata_peaks_ltd_ = self.findpeaks_radh(t_st, t_ed, i_ch, timedata[idx_time_st:idx_time_ed], voltdata_array, isCondAV=True, isPlot=False, Mode_find_peaks='avalanche20240403', prom_min=prom_min, prom_max=prom_max)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        if arr_peak_selection == None:
            timedata_peaks_ltd = timedata_peaks_ltd_
        else:
            timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 15e-3
        t_ed_condAV = 25e-3#4e-3#6e-3#10e-3#
        for i in range(len(timedata_peaks_ltd)):
            if i == 0:
                if Mode_ece_radhpxi_calThom:
                    buf_radh = voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed],
                                                                   timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                else:
                    buf_radh = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed],
                                                           timedata_peaks_ltd[i] - t_st_condAV):find_closest(
                        timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i] + t_ed_condAV), :]
                buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i] + timedata[idx_time_st] - timedata[0]
                buf_t_nel = t_nel[find_closest( t_nel, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_nel, timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]

                buf_nel = data_fir_nel[find_closest( t_nel, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_nel, timedata_peaks_ltd[i]+t_ed_condAV),:]

            else:
                try:
                    buf_radh_ = voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    if Mode_ece_radhpxi_calThom:
                        buf_radh += voltdata_array_calThom[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    else:
                        buf_radh += voltdata_array[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV), :]
                    buf_t_radh = timedata[find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]-t_st_condAV):find_closest(timedata[idx_time_st:idx_time_ed], timedata_peaks_ltd[i]+t_ed_condAV)] - timedata_peaks_ltd[i]+ timedata[idx_time_st] -timedata[0]
                    buf_t_nel = t_nel[find_closest( t_nel, timedata_peaks_ltd[i] - t_st_condAV):find_closest( t_nel,
                                                                                                                      timedata_peaks_ltd[
                                                                                                                          i] + t_ed_condAV)] - \
                                timedata_peaks_ltd[i]
                    buf_nel += data_fir_nel[find_closest( t_nel, timedata_peaks_ltd[i]-t_st_condAV):find_closest( t_nel, timedata_peaks_ltd[i]+t_ed_condAV),:]
                except Exception as e:
                    print(e)

        cond_av_radh = buf_radh/len(timedata_peaks_ltd)
        cond_av_nel = buf_nel/len(timedata_peaks_ltd)
        #cond_av_nel = np.array(normalize_all_columns_np(cond_av_nel_))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        label_radh = "ECE(rho=" + str(rho[i_ch]) + ")"
        c = ax1.plot(buf_t_radh, cond_av_radh[:, i_ch], 'r-', label=label_radh, zorder=2)
        #a = ax2.plot( buf_t_nel, cond_av_nel[:], 'g-', label=data_nel.getValName(num_val), zorder=2)
        for i in range(vnum):
            a = ax2.plot( buf_t_nel, cond_av_nel[:,i] + 1.2*i, '-', zorder=2)
        #a = ax2.plot( buf_t_qe, cond_av_qe[:, 18], '-', label=data_qe.getValName(num_val)+"18", zorder=2)
        #a = ax2.plot( buf_t_qe, cond_av_qe[:, 20], '-', label=data_qe.getValName(num_val)+"20", zorder=2)
        plt.title('Prominence:%.1fto%.1f, #%d' % (prom_min, prom_max, self.ShotNo), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel(data_nel.getValName(num_val) + " [" + data_nel.getValUnit(num_val) + "]", rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        #ax1.set_ylim(-0.5, 0.1)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.1fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()
        plt.close()



        return buf_t_radh, cond_av_radh, buf_t_nel, R_fir_mm, cond_av_nel,data_nel.getValName(num_val), data_nel.getValUnit(num_val), rho, len(timedata_peaks_ltd), fs

    def ana_plot_ece(self, isSaveData=False):
        data = AnaData.retrieve('ece_fast', self.ShotNo, 1)
        #data = AnaData.retrieve('ece_fast_calThom', self.ShotNo, 1)

        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
        #rho = data.getValData('rho_vacuum')
        #print(rho[0])
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        #t = data.getDimData('time')
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        #Te = data.getValData('Te_notch_LP200Hz')
        #Te = data.getValData('Te_notch')
        Te = data.getValData('Te')
        #rho = data.getValData('rho_vacuum')

        data = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        '''
        fig = plt.figure(figsize=(5, 2.5), dpi=250)
        plt.title(self.ShotNo, fontsize=12, loc='right')
        arr_erace = [5, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,32,33,44,45,46,47,50,51]#,58]
        #plt.pcolormesh(t[:-1], r, (Te[1:, :]-Te[:-1, :]).T, cmap='bwr')
        #plt.pcolormesh(t, r[arr_erace], Te[:,arr_erace].T, cmap='jet', vmin=0, vmax=3)
        plt.pcolormesh(t, rho[0,arr_erace], Te[:,arr_erace].T, cmap='jet', vmin=0, vmax=4)
        #plt.pcolormesh(t, r, Te.T, cmap='jet', vmax=6)
        #plt.pcolormesh(t, rho[0,:], Te.T, cmap='jet', vmax=6)
        #plt.pcolormesh(Te.T, cmap='jet', vmax=6)
        #plt.pcolormesh(t, r[1:-1], (Te[:, 2:]-Te[:, 1:-1]-Te[:, 1:-1]+Te[:, :-2]).T, cmap='bwr')
        cb = plt.colorbar(pad=0.01)
        #cb.set_label("d2Te/dr2")
        cb.set_label(r'$T_e [keV]$', rotation=270, labelpad=16, fontsize=12)
        plt.xlabel(r"Time [sec]", fontsize=12)
        #plt.ylabel("R [m]")
        plt.ylabel(r"r/a(vacuum)", fontsize=12)
        plt.xlim(3.0, 6.0)
        plt.ylim(0.38, 0.7)
        #plt.ylim(3.85, 4.2)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()
        '''

        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t_fir = data.getDimData('Time')

        f, axarr = plt.subplots(2, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 6), dpi=150, sharex=True)
        #for i in range(30, 67):
        #    plt.plot(t, Te[:, i], label=str(i))
        #    plt.legend()
        #    plt.xlim(4.3, 4.8)
        #    plt.show()

        axarr[0].set_title(self.ShotNo, fontsize=18, loc='right')
        #axarr[0].set_ylabel(r'$T_e (ECE)$', fontsize=18)
        axarr[1].set_ylabel(r'$T_e$ [keV]', fontsize=18)
        axarr[1].set_xlabel('Time [sec]', fontsize=18)
        axarr[0].set_ylabel(r'$n_eL\ [10^{19}m^{-2}]$', fontsize=14)
        axarr[1].set_ylim(0, 14)
        #axarr[0].set_xlim(4.3, 4.8)
        #axarr[0].set_xlim(2.8, 6.0)
        #axarr[0].set_xlim(3.3, 5.3)
        #axarr[1].set_xlim(4.35, 4.55)
        #axarr[0].set_xlim(4.4, 4.6)
        axarr[0].set_xlim(3, 6.5)
        #axarr[1].set_xlim(4.4, 4.6)    #timerange
        #axarr[0].set_ylim(0, )
        axarr[1].set_ylim(0, 4)
        #axarr[0].set_xticklabels([])
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)

        #arr_erace = [5, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]#,22,24,25,26,27,28,32,33,44,45,46,47,50,51]#,58]
        #axarr[1].plot(t, Te[:, arr_erace])
        axarr[1].plot(t, Te)
        for i in range(vnum):
            ne_bar = data.getValData(i)
            axarr[0].plot(t_fir, ne_bar)
        f.align_labels()
        plt.savefig("timetrace_ECE_FIR_No%d.png" % self.ShotNo)
        plt.show()

        t_common = np.linspace(3.3, 9.3, 6000)
        i_ch=65
        ne_bar = data.getValData(6)
        interp_Te = interp1d(t, Te[:,i_ch])
        interp_ne = interp1d(t_fir, ne_bar)
        Te_interpolated = interp_Te(t_common)
        ne_interpolated = interp_ne(t_common)

        #plt.plot(t_common, Te_interpolated)
        #plt.show()
        #plt.plot(t_common, ne_interpolated)
        #plt.show()
        '''
        plt.plot(ne_interpolated, Te_interpolated)
        i_ch=0
        interp_Te = interp1d(t, Te[:, i_ch])
        Te_interpolated = interp_Te(t_common)
        label_rho = rho[0, i_ch]
        plt.plot(ne_interpolated, Te_interpolated, label=label_rho)
        i_ch=10
        interp_Te = interp1d(t, Te[:,i_ch])
        Te_interpolated = interp_Te(t_common)
        label_rho = rho[0, i_ch]
        plt.plot(ne_interpolated, Te_interpolated, label=label_rho)
        i_ch=20
        interp_Te = interp1d(t, Te[:,i_ch])
        Te_interpolated = interp_Te(t_common)
        label_rho = rho[0, i_ch]
        plt.plot(ne_interpolated, Te_interpolated, label=label_rho)
        i_ch=30
        interp_Te = interp1d(t, Te[:,i_ch])
        Te_interpolated = interp_Te(t_common)
        label_rho = rho[0, i_ch]
        plt.plot(ne_interpolated, Te_interpolated, label=label_rho)
        i_ch=40
        interp_Te = interp1d(t, Te[:,i_ch])
        Te_interpolated = interp_Te(t_common)
        label_rho = rho[0, i_ch]
        plt.plot(ne_interpolated, Te_interpolated, label=label_rho)
        i_ch=60
        interp_Te = interp1d(t, Te[:,i_ch])
        Te_interpolated = interp_Te(t_common)
        label_rho = rho[0, i_ch]
        plt.plot(ne_interpolated, Te_interpolated, label=label_rho)
        plt.title(self.ShotNo, fontsize=14, loc='right')
        plt.xlabel(r'$n_eL\ [10^{19}m^{-2}]$', fontsize=14)
        plt.ylabel("Te[keV]", fontsize=14)
        plt.xlim(0, 3)
        plt.ylim(0,7)
        plt.legend()
        plt.savefig("Te_vs_neL_No%d.png" % self.ShotNo)
        plt.show()
        '''

        if isSaveData:
            #t_data_ece = np.vstack((t, Te[:, arr_erace].T))
            t_data_ece = np.vstack((t, Te.T))
            filename = '_%d.txt' % (self.ShotNo)
            np.savetxt('time_ECE' + filename, t_data_ece.T)


    def ana_plot_discharge_waveform_basic(self):
        data_wp = AnaData.retrieve('wp', self.ShotNo, 1)
        data_ECH = AnaData.retrieve('echpw', self.ShotNo, 1)
        data_fir = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        t_ECH = data_ECH.getDimData('Time')
        t_Wp = data_wp.getDimData('Time')
        t_fir = data_fir.getDimData('Time')

        plt.clf()
        f, axarr = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(7,5), dpi=300, sharex=True)

        # データ取得
        title = ('#%d') % (self.ShotNo)
        vnum_ECH = data_ECH.getValNo()
        axarr[0].set_title(title, loc='left', fontsize=16)
        view_ECH = np.array([12, 0, 1, 2, 4, 5, 6, 7])
        for i in range(len(view_ECH)):
            data_ = data_ECH.getValData(view_ECH[i])
            axarr[3].plot(t_ECH, data_, '-', label=data_ECH.getValName(view_ECH[i]), color=self.arr_color[i])

        arr_data_name = ['nb1', 'nb2', 'nb3', 'nb4a', 'nb4b', 'nb5a', 'nb5b']
        for i in range(len(arr_data_name)):
            data_NBI = AnaData.retrieve(arr_data_name[i] + 'pwr_temporal', self.ShotNo, 1)
            data_NBI_ = data_NBI.getValData(1)
            t_NBI = data_NBI.getDimData('time')
            label = str.upper(arr_data_name[i])
            axarr[2].plot(t_NBI, data_NBI_, label=label, color=self.arr_color[i])

        data_wp_ = data_wp.getValData('Wp')
        data_fir_ = data_fir.getValData(0)
        axarr[0].plot(t_Wp, data_wp_, color='black')
        axarr[1].plot(t_fir, data_fir_, color='black')
        axarr[0].set_xlim(3,6)
        axarr[0].set_ylim(0,)
        axarr[1].set_ylim(0,)
        axarr[2].set_ylim(0,)
        axarr[3].set_ylim(0,)
        axarr[2].legend(frameon=False, loc='right', fontsize=5)
        axarr[3].legend(frameon=False, loc='right', fontsize=5)
        axarr[0].set_ylabel('Wp [kJ]', fontsize=11)
        axarr[1].set_ylabel(r'$\overline{n_e}\ [10^{19} m^{-3}$]', fontsize=10)
        axarr[2].set_ylabel('NBI [MW]', fontsize=11)
        axarr[3].set_ylabel('ECH [MW]', fontsize=11)
        axarr[3].set_xlabel('Time [sec]', fontsize=11)
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        f.align_labels()
        plt.plot()
        plt.show()
        #plt.savefig("timetrace_%d.png" % self.ShotNo)

    def plot_NBI_ASTI(self):

        data_name_NBEL = 'NBEL'
        i_ch = 54#57
        timedata = TimeData.retrieve(data_name_NBEL, self.ShotNo, 1, 1).get_val()
        voltdata = VoltData.retrieve(data_name_NBEL, self.ShotNo, 1, i_ch+1).get_val()

        plt.plot(timedata, voltdata, label='NB#4A_')
        i_ch = 55#60
        timedata = TimeData.retrieve(data_name_NBEL, self.ShotNo, 1, 1).get_val()
        voltdata = VoltData.retrieve(data_name_NBEL, self.ShotNo, 1, i_ch+1).get_val()

        plt.plot(timedata, voltdata, label='NB#4B_')

        i_ch = 56#57
        timedata = TimeData.retrieve(data_name_NBEL, self.ShotNo, 1, 1).get_val()
        voltdata = VoltData.retrieve(data_name_NBEL, self.ShotNo, 1, i_ch+1).get_val()

        plt.plot(timedata, voltdata, label='NB#5A')
        i_ch = 59#60
        timedata = TimeData.retrieve(data_name_NBEL, self.ShotNo, 1, 1).get_val()
        voltdata = VoltData.retrieve(data_name_NBEL, self.ShotNo, 1, i_ch+1).get_val()

        plt.plot(timedata, voltdata, label='NB#5B')
        plt.xlabel('Time [s]')
        plt.ylabel('NBEL')
        plt.legend()
        plt.title('#' + str(self.ShotNo))
        plt.show()

    def ana_plot_discharge_waveform4ITB(self, isSaveData=False):
        data_wp = AnaData.retrieve('wp', self.ShotNo, 1)
        data_ECH = AnaData.retrieve('echpw', self.ShotNo, 1)
        data_fir = AnaData.retrieve('fir_nel', self.ShotNo, 1)
        data_ip = AnaData.retrieve('ip', self.ShotNo, 1)
        data_shotinfo = AnaData.retrieve('shotinfo', self.ShotNo, 1)
        valno = data_shotinfo.getValNo()
        data_array_shotinfo = np.zeros(valno)
        for i in range(valno):
            data_array_shotinfo[i] = data_shotinfo.getValData(i)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        t_ECH = data_ECH.getDimData('Time')
        t_Wp = data_wp.getDimData('Time')
        t_fir = data_fir.getDimData('Time')
        t_ip = data_ip.getDimData('Time')


        f, axarr = plt.subplots(5, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(7,7), dpi=150, sharex=True)

        # データ取得
        title_shotinfo = (r'$B=%.3fT, R_{ax}=%.3fm, \gamma =%.4f, B_q=%.1f$') % (data_array_shotinfo[7], data_array_shotinfo[8], data_array_shotinfo[9], data_array_shotinfo[10])
        #title_shotinfo_1 = r'$B={B:.3f}T, $'.format(B=data_array_shotinfo[7])
        #title_shotinfo_2 = r'$B={Rax_vac:.3f}m, \gamma={gamma:.4f}, B_q={Bq:.1f}%$'.format(Rax_vac=data_array_shotinfo[8], gamma=data_array_shotinfo[9], Bq=data_array_shotinfo[10])
        #title_shotinfo = title_shotinfo_1 + r'$B_{ax}$' + title_shotinfo_2 + '%'
        title_date = '{year:d}-{month:02,d}-{day:02,d} {hour:d}:{minute:02,d}'.format(year=int(data_array_shotinfo[2]), month=int(data_array_shotinfo[3]), day=int(data_array_shotinfo[4]), hour=int(data_array_shotinfo[5]), minute=int(data_array_shotinfo[6]))
        title = ('#%d') % (self.ShotNo)
        vnum_ECH = data_ECH.getValNo()
        axarr[0].set_title(title, loc='left', fontsize=16)
        axarr[0].set_title(title_date + '\n' + title_shotinfo, loc='right', fontsize=12)
        axarr[2].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        view_ECH = np.array([12, 0, 1, 2, 4, 6])
        #view_ECH = np.array([12, 0, 1, 2, 4, 5, 6, 7])

        data_ECH_array = np.zeros((len(t_ECH), len(view_ECH)+1))
        data_ECH_array[:, 0] = t_ECH
        for i in range(len(view_ECH)):
            data_ = data_ECH.getValData(view_ECH[i])
            axarr[4].plot(t_ECH, data_, '-', label=data_ECH.getValName(view_ECH[i]), color=self.arr_color[i])
            data_ECH_array[:, i+1] = data_

        arr_data_name = ['nb1', 'nb2', 'nb3', 'nb4a', 'nb4b', 'nb5a', 'nb5b']
        data_NBI = AnaData.retrieve(arr_data_name[0] + 'pwr_temporal', self.ShotNo, 1)
        t_NBI = data_NBI.getDimData('time')
        data_NBI_array = np.zeros((len(t_NBI), len(arr_data_name)+1))
        data_NBI_array[:, 0] = t_NBI
        for i in range(len(arr_data_name)):
            data_NBI = AnaData.retrieve(arr_data_name[i] + 'pwr_temporal', self.ShotNo, 1)
            t_NBI = data_NBI.getDimData('time')
            data_NBI_ = data_NBI.getValData(1)
            label = str.upper(arr_data_name[i])
            axarr[3].plot(t_NBI, data_NBI_, label=label, color=self.arr_color[i])
            data_NBI_array[:, i+1] = data_NBI_

        data_wp_ = data_wp.getValData('Wp')
        data_fir_ = data_fir.getValData(0)
        data_ip_ = data_ip.getValData('Ip')
        axarr[0].plot(t_Wp, data_wp_, color='black')
        axarr[1].plot(t_fir, data_fir_, color='red')
        axarr[2].plot(t_ip, data_ip_, label='Ip', color='blue')
        axarr[0].set_xlim(3, 9)    #timerange
        axarr[0].set_ylim(0,)
        axarr[1].set_ylim(0,)
        axarr[2].set_ylim(-5,15)
        axarr[3].set_ylim(0,)
        axarr[4].set_ylim(0,)
        axarr[3].legend(frameon=False, loc='right', fontsize=7)
        axarr[4].legend(frameon=False, loc='right', fontsize=7)
        axarr[0].set_ylabel('Wp [kJ]', fontsize=11)
        axarr[1].set_ylabel(r'$\overline{n_e}\ [10^{19} m^{-3}$]', fontsize=10)
        axarr[2].set_ylabel('Ip [kA]', fontsize=11)
        axarr[3].set_ylabel('NBI [MW]', fontsize=11)
        axarr[4].set_ylabel('ECH [MW]', fontsize=11)
        axarr[4].set_xlabel('Time [sec]', fontsize=11)
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        axarr[4].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        f.align_labels()
        plt.savefig("timetrace_%d.png" % self.ShotNo)
        plt.show()
        if isSaveData:
            t_data_Wp = np.stack((t_Wp, data_wp_))
            t_data_fir = np.stack((t_fir, data_fir_))
            t_data_Ip = np.stack((t_ip, data_ip_))
            filename = '_%d.txt' % (self.ShotNo)
            np.savetxt('time_Wp' + filename, t_data_Wp.T)
            np.savetxt('time_fir' + filename, t_data_fir.T)
            np.savetxt('time_ip' + filename, t_data_Ip.T)
            np.savetxt('time_ECH' + filename, data_ECH_array)
            np.savetxt('time_NBI' + filename, data_NBI_array)

    def ana_plot_discharge_waveform(self):
        data_wp = AnaData.retrieve('wp', self.ShotNo, 1)
        data_ECH = AnaData.retrieve('echpw', self.ShotNo, 1)
        data_ip = AnaData.retrieve('ip', self.ShotNo, 1)
        data_fir = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        t_ECH = data_ECH.getDimData('Time')
        t_Wp = data_wp.getDimData('Time')
        t_ip = data_ip.getDimData('Time')
        t_fir = data_fir.getDimData('Time')
        #dnum = data.getDimNo()
        #for i in range(dnum):
        #    print(i, data.getDimName(i), data.getDimUnit(i))
        #vnum = data.getValNo()
        #for i in range(vnum):
        #    print(i, data.getValName(i), data.getValUnit(i))

        plt.clf()
        f, axarr = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(10,6), dpi=300, sharex=True)

        #for i, ax in enumerate(f.axes):
        #    ax.grid('on', linestyle='--')
        #    if i % len(t) != 0:
        #        ax.set_yticklabels([])
        #    if i < 2*len(t):
        #        ax.set_xticklabels([])
        # データ取得
        title = ('#%d') % (self.ShotNo)
        vnum_ECH = data_ECH.getValNo()
        axarr[0].set_title(title, loc='left', fontsize=16)
        view_ECH = np.array([12, 0, 1, 2, 4, 5, 6, 7])
        axarr[2].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        for i in range(len(view_ECH)):
            data_ = data_ECH.getValData(view_ECH[i])
            axarr[1].plot(t_ECH, data_, '-', label=data_ECH.getValName(view_ECH[i]))

        arr_data_name = ['nb1', 'nb2', 'nb3', 'nb4a', 'nb4b', 'nb5a', 'nb5b']
        for i in range(len(arr_data_name)):
            data_NBI = AnaData.retrieve(arr_data_name[i] + 'pwr_temporal', self.ShotNo, 1)
            data_NBI_ = data_NBI.getValData(1)
            t_NBI = data_NBI.getDimData('time')
            label = str.upper(arr_data_name[i])
            axarr[0].plot(t_NBI, data_NBI_, label=label)

        data_wp_ = data_wp.getValData('Wp')
        data_ip_ = data_ip.getValData('Ip')
        data_fir_ = data_fir.getValData(0)
        axarr[2].plot(t_Wp, data_wp_, label='Wp')
        axarr[2].plot(t_ip, 10*data_ip_, label='Ip')
        axarr[2].plot(t_fir, 100*data_fir_, label='ne_bar')
        axarr[0].set_xlim(3,6)
        axarr[0].legend(frameon=False, loc='right')
        axarr[1].legend(frameon=False, loc='right')
        axarr[2].legend(frameon=False, loc='right')
        axarr[0].set_ylabel('NBI [MW]', fontsize=16)
        axarr[1].set_ylabel('ECH [MW]', fontsize=16)
        axarr[2].set_xlabel('Time [sec]', fontsize=16)
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        plt.plot()
        plt.show()

    def ana_plot_discharge_waveforms(self, arr_ShotNo):
        plt.clf()
        f, axarr = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(7,5), dpi=150, sharex=True)
        axarr[2].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        for i in range(len(arr_ShotNo)):
           data_wp = AnaData.retrieve('wp', arr_ShotNo[i], 1)
           data_fir = AnaData.retrieve('fir_nel', arr_ShotNo[i], 1)
           #data_ECH = AnaData.retrieve('echpw', self.ShotNo, 1)
           data_ip = AnaData.retrieve('ip', arr_ShotNo[i], 1)
           #data_fir = AnaData.retrieve('fir_nel', self.ShotNo, 1)

           # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
           t_Wp = data_wp.getDimData('Time')
           t_ip = data_ip.getDimData('Time')
           t_fir = data_fir.getDimData('Time')
           data_wp_ = data_wp.getValData('Wp')
           data_ip_ = data_ip.getValData('Ip')
           data_fir_ = data_fir.getValData('ne_bar(3669)')
           axarr[0].plot(t_Wp, data_wp_, label=arr_ShotNo[i])
           axarr[1].plot(t_fir, data_fir_, label=arr_ShotNo[i])
           axarr[2].plot(t_ip, data_ip_, label=arr_ShotNo[i])
        title = str(arr_ShotNo)
        axarr[0].set_title('#' + title[1:-1], loc='left')


        axarr[0].set_xlim(3,6)
        axarr[0].set_ylim(0,)
        axarr[1].set_ylim(0,)
        axarr[0].legend(frameon=False, loc='right', fontsize=7)
        axarr[1].legend(frameon=False, loc='right', fontsize=7)
        axarr[2].legend(frameon=False, loc='right', fontsize=7)
        axarr[0].set_ylabel('Wp [kJ]', fontsize=11)
        axarr[1].set_ylabel(r'$\overline{n_e}\ [10^{19} m^{-3}$]', fontsize=11)
        axarr[2].set_ylabel('Ip [kA]', fontsize=11)
        axarr[2].set_xlabel('Time [sec]', fontsize=11)
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        f.align_labels()
        plt.plot()
        plt.show()

    def plot_2TS(self):
        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        r = data.getDimData('R')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        #kernel = np.full((5,5), 1/25)
        #kernel = np.full((3,3), 1/9)
        kernel = np.full((1,1), 1/1)
        #arr_all = movingaverage(data_2d=Te, axis=0, windowsize=5)
        arr_all = signal.convolve2d(Te, kernel, boundary='symm', mode='same')
        arr_all_ne = signal.convolve2d(ne, kernel, boundary='symm', mode='same')
        arr_all_dTe = signal.convolve2d(dTe, kernel, boundary='symm', mode='same')
        arr_all_dne = signal.convolve2d(dne, kernel, boundary='symm', mode='same')
        arr_R = r
        arr_1 = reff
        arr_dR = arr_R[1:] - arr_R[:-1]
        arr_dR_s = np.squeeze(arr_dR)
        arr_2 = (arr_all[:,1:] - arr_all[:, :-1]).T
        arr_dTedR = arr_2.T/arr_dR_s

        fig = plt.figure(figsize=(5, 2.5), dpi=150)
        plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
        i_time = 45#44
        #label = 'SN'+str(self.ShotNo) + '(t=' + str(np.array(t[i_time])) + 's)'
        label = 'D (SN'+str(self.ShotNo) + ')'
        #plt.fill_between(reff[i_time, :], arr_all[i_time, :] - arr_all_dTe[i_time, :],
        #                 arr_all[i_time, :] + arr_all_dTe[i_time, :], alpha=0.15, color='blue', edgecolor='white')
        #plt.plot(reff[i_time, :], arr_all[i_time, :], 'b-', label=label, linewidth=1)
        #plt.errorbar(reff[i_time], arr_all[i_time, :], yerr=arr_all_dTe[i_time, :], capsize=4, fmt='ro', markersize=4, label=label)
        plt.errorbar(reff[i_time], arr_all_ne[i_time, :], yerr=arr_all_dne[i_time, :], capsize=4, fmt='ro', markersize=4, label=label)

        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        r = data.getDimData('R')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        arr_all = signal.convolve2d(Te, kernel, boundary='symm', mode='same')
        arr_all_ne = signal.convolve2d(ne, kernel, boundary='symm', mode='same')
        arr_all_dTe = signal.convolve2d(dTe, kernel, boundary='symm', mode='same')
        arr_all_dne = signal.convolve2d(dne, kernel, boundary='symm', mode='same')
        arr_R = r
        arr_1 = reff
        arr_dR = arr_R[1:] - arr_R[:-1]
        arr_dR_s = np.squeeze(arr_dR)
        arr_2 = (arr_all[:,1:] - arr_all[:, :-1]).T
        arr_dTedR = arr_2.T/arr_dR_s

        i_time = 42
        #label = 'SN'+str(self.ShotNo_2) + '(t=' + str(np.array(t[i_time])) + 's)'
        label = 'H (SN'+str(self.ShotNo_2) + ')'
        #plt.fill_between(reff[i_time, :], arr_all[i_time, :] - arr_all_dTe[i_time, :],
        #                 arr_all[i_time, :] + arr_all_dTe[i_time, :], alpha=0.15, color='red', edgecolor='white')
        #plt.plot(reff[i_time, :], arr_all[i_time, :], 'r-', label=label, linewidth=1)
        #plt.errorbar(reff[i_time], arr_all[i_time, :], yerr=arr_all_dTe[i_time, :], capsize=4, fmt='b^', markersize=4, label=label)
        plt.errorbar(reff[i_time], arr_all_ne[i_time, :], yerr=arr_all_dne[i_time, :], capsize=4, fmt='b^', markersize=4, label=label)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        ## plt.plot(hs_thomson['Time'], arr_all[:, 13:15], '-')
        plt.legend(fontsize=8)
        #plt.ylim(0, 15)
        plt.ylim(0, 1.5)
        plt.xlim(-1.0, 1.0)
        #plt.xlim(-1.2, 1.2)
        #plt.xlim(0, 1.0)
        plt.xlabel('$r_{eff}/a_{99}$')
        #plt.ylabel('$T_e [keV]$')
        plt.ylabel('$n_e [10^{19}m^{-3}]$')
        plt.title('SN' + str(self.ShotNo) + ',' + str(self.ShotNo_2), loc='right')
        plt.tight_layout()
        plt.show()

    def calc_qe(self):
        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reffa99 = data.getValData('reff/a99')
        reff = data.getValData('reff')
        dVdreff = data.getValData('dVdreff')
        ne_fit = data.getValData('ne_fit')
        t_ece, reffa99_ece, cond_av_ece = self.condAV_radh_highK_mp_trigLowPassRadh_multishots(t_st=4.0, t_ed=6, prom_min=0.4, prom_max=10.0)
        dTedt = (cond_av_ece[1:,:]-cond_av_ece[:-1,:])/(t_ece[1]-t_ece[0])

        i_reff_integ_start = 6
        qe = np.zeros(np.shape(dTedt[:, i_reff_integ_start:]))
        dqedr = np.zeros(np.shape(dTedt[:, i_reff_integ_start:]))

        fig = plt.figure(figsize=(8,6), dpi=150)
        #title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        #plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*t_ece[1:], reffa99_ece, dTedt.T, cmap='bwr', vmin=-10, vmax=10)
        cb = plt.colorbar()
        cb.set_label("dTe/dt [0,1]", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        plt.show()
        plt.close()

        f = interp1d(reffa99[52], dVdreff[52], kind='cubic')
        dVdreff_ece = f(reffa99_ece)
        f2 = interp1d(reffa99[52], reff[52], kind='cubic')
        reff_ece = f2(reffa99_ece)
        f3 = interp1d(reffa99[52], ne_fit[52], kind='cubic')
        ne_fit_ece = f3(reffa99_ece)

        for i in range(len(t_ece)-i_reff_integ_start):

            qe[i,:] = -(3/2)*integrate.cumtrapz(ne_fit_ece[i_reff_integ_start:]*dTedt[i,i_reff_integ_start:]*dVdreff_ece[i_reff_integ_start:]*(reff_ece[i_reff_integ_start:]-reff_ece[i_reff_integ_start-1:-1]), reff_ece[i_reff_integ_start:], initial=0)/(dVdreff_ece[i_reff_integ_start:]*ne_fit_ece[i_reff_integ_start:])
            dqedr[i,:] = ne_fit_ece[i_reff_integ_start:]*dTedt[i,i_reff_integ_start:]*dVdreff_ece[i_reff_integ_start:]*(reffa99_ece[i_reff_integ_start:]-reffa99_ece[i_reff_integ_start-1:-1])

        fig = plt.figure(figsize=(8,6), dpi=150)
        #title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        #plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*t_ece[1:], reffa99_ece[i_reff_integ_start:], dqedr.T, cmap='bwr')#, vmin=-20, vmax=20)
        cb = plt.colorbar()
        cb.set_label("dqe/dreff/ne", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(8,6), dpi=150)
        #title = 'Low-pass(<%dkHz), Prominence:%.1fto%.1f, #%d-%d' % (fs/1e3, prom_min, prom_max, shot_st, shot_ed)
        #plt.title(title, fontsize=16, loc='right')
        plt.pcolormesh(1e3*t_ece[1:], reffa99_ece[i_reff_integ_start:], qe.T, cmap='bwr')#, vmin=-0.050, vmax=0.05)
        cb = plt.colorbar()
        cb.set_label("qe/ne", fontsize=18)
        plt.xlabel("Time [ms]", fontsize=18)
        plt.ylabel(r"$r_{eff}/a_{99}$", fontsize=18)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        plt.show()
        plt.close()

        '''
        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        plt.plot(reffa99[10], dVdreff[10])
        plt.plot(reffa99_ece, dVdreff_ece)
        plt.show()
        '''
        plt.plot(reffa99[52], ne_fit[52])
        plt.plot(reffa99_ece, ne_fit_ece)
        plt.show()

    def plot_TS(self):
        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        Te = data.getValData('Te')
        dTe = data.getValData('dTe')
        #dTedr = data.getValData('dTedr')
        #dTedr_err = data.getValData('dTedr_err')
        ne = data.getValData('ne_calFIR')
        dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        r = data.getDimData('R')


        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        #kernel = np.full((5,5), 1/25)
        kernel = np.full((3,3), 1/9)
        #kernel = np.full((1,1), 1/1)
        #arr_all = movingaverage(data_2d=Te, axis=0, windowsize=5)
        arr_all = signal.convolve2d(Te, kernel, boundary='symm', mode='same')
        arr_all_dTe = signal.convolve2d(dTe, kernel, boundary='symm', mode='same')
        arr_all_ne = signal.convolve2d(ne, kernel, boundary='symm', mode='same')
        arr_all_dne = signal.convolve2d(dne, kernel, boundary='symm', mode='same')
        arr_all_Pe_err = arr_all*arr_all_dne + arr_all_ne*arr_all_dTe
        #arr_all_dTedr = signal.convolve2d(dTedr, kernel, boundary='symm', mode='same')
        #arr_all_dTedr_err = signal.convolve2d(dTedr_err, kernel, boundary='symm', mode='same')
        arr_R = r
        arr_1 = reff
        arr_dR = arr_R[1:] - arr_R[:-1]
        arr_dR_s = np.squeeze(arr_dR)
        arr_all_pe = 1.602176634*arr_all*arr_all_ne
        arr_2_pe = (arr_all_pe[:,1:] - arr_all_pe[:, :-1]).T
        arr_2_pe_err = (arr_all_Pe_err[:,1:] + arr_all_Pe_err[:, :-1]).T
        arr_dPedR = arr_2_pe.T/arr_dR_s
        arr_dPedR_err = arr_2_pe_err.T
        arr_2 = (arr_all[:,1:] - arr_all[:, :-1]).T
        arr_dTedR = arr_2.T/arr_dR_s
        #arr_dTedR = arr_all_dTedr[:,:-1]


        fig = plt.figure(figsize=(8,2), dpi=600)
        plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加

        X, _ = np.meshgrid(t, reff[0, :])
        #plt.pcolormesh(hs_thomson['Time'], hs_thomson['R'], arr_all.T, cmap='jet', vmin=0, vmax=6)
        #plt.pcolormesh(X, arr_1.T, arr_all.T, cmap='jet', vmin=0, vmax=10)
        #plt.pcolormesh((arr_all[:,1:] - arr_all[:, :-1]).T, cmap='bwr', vmin=-1, vmax=1)
        #plt.pcolormesh(X[1:, :], arr_1[:, 1:].T, arr_2, cmap='bwr', vmin=-1, vmax=1)
        #plt.pcolormesh(X[1:, :], arr_1[:, 1:].T, arr_dTedR.T, cmap='bwr', vmin=-100, vmax=100)
        plt.pcolormesh(X[1:, 2:69], arr_1[2:69, 1:].T, arr_dPedR[2:69, :].T, cmap='bwr', vmin=-150, vmax=150)
        #plt.pcolormesh(X, arr_1.T, arr_all.T, cmap='jet', vmin=0, vmax=15)
        #plt.pcolormesh(np.array(tsmap_calib['Time']), np.array(tsmap_calib['reff/a99']), (arr_all[:,1:] - arr_all[:, :-1]).T, cmap='bwr', vmin=-1, vmax=1)
        #plt.vlines(4.398907, -1, 1, colors='red')
        #plt.vlines(4.399857, -1, 1, colors='blue')
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.xlabel('Time [sec]')
        #plt.ylim(-1, 1)
        plt.ylabel('R [m]')
        #plt.ylabel('$r_{eff}/a_{99}$', fontsize=15)
        #plt.ylim(2.9, 4.50)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.get_yaxis().labelpad = 15
        #cbar.set_label('$dT_e/dR$ [keV/m]', rotation=270)
        #cbar.set_label('$T_e$ [keV]', rotation=270)
        idx_time = np.arange(len(t))
        idx_max = np.argmax(arr_dPedR[:, 37:53], axis=1)
        #idx_max = np.argmax(arr_dTedR[:, 30:70], axis=1)
        #idx_min = np.argmin(arr_dTedR[:, 30:70], axis=1)
        idx_min = np.argmin(arr_dPedR[:, 53:67], axis=1)
        #reffa99_max = arr_1[idx_time, idx_max+31]
        reffa99_max = arr_1[idx_time, idx_max+38]
        reffa99_min = arr_1[idx_time, idx_min+54]
        plt.plot(t[2:69], reffa99_max[2:69], 'ro', markersize=2)
        plt.plot(t[2:69], reffa99_min[2:69], 'bo', markersize=2)
        plt.xlim(3.0, 6.0)
        #plt.ylim(-1, 1)
        plt.ylim(-0.6, 0.6)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        #plt.show()
        plt.show()

        '''
        fig = plt.figure(figsize=(5, 2.5), dpi=150)
        #i_time = 29
        #label = 't=' + str(np.array(t[i_time])) + 's'
        #plt.fill_between(reff[i_time, :], arr_all[i_time, :] - arr_all_dTe[i_time, :],
        #                 arr_all[i_time, :] + arr_all_dTe[i_time, :], alpha=0.15, color='blue', edgecolor='white')
        #plt.plot(reff[i_time, :], arr_all[i_time, :], '-', label=label, linewidth=1, color='blue')
        ##plt.errorbar(reff[i_time], arr_all[i_time, :], yerr=arr_all_dTe[i_time, :], capsize=2, fmt='bo', markersize=2)
        i_time = 45#44
        label = 't=' + str(np.array(t[i_time])) + 's'
        plt.fill_between(reff[i_time, :], arr_all[i_time, :] - arr_all_dTe[i_time, :],
                         arr_all[i_time, :] + arr_all_dTe[i_time, :], alpha=0.15, color='red', edgecolor='white')
        plt.plot(reff[i_time, :], arr_all[i_time, :], 'r-', label=label, linewidth=1)
        #plt.errorbar(reff[i_time], arr_all[i_time, :], yerr=arr_all_dTe[i_time, :], capsize=2, fmt='ro', markersize=2, label=label)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        ## plt.plot(hs_thomson['Time'], arr_all[:, 13:15], '-')
        plt.legend(fontsize=8)
        plt.ylim(0, 20)
        plt.xlim(-1.0, 1.0)
        #plt.xlim(0.2, 1.2)
        #plt.xlim(1, 1.0)
        plt.xlabel('$r_{eff}/a_{100}$')
        plt.ylabel('$T_e [keV]$')
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.show()
        '''

        fig = plt.figure(figsize=(9,2), dpi=150)
        #plt.plot(t, np.max(arr_dTedR[:, 31:90], axis=1), 'r-', label='reff/a99<0')
        #plt.plot(t, np.max(arr_dPedR[:, 31:70], axis=1), 'r-', label='reff/a99<0')
        plt.plot(t, np.abs(arr_dPedR[idx_time, idx_max+38]), 'r-', label='reff/a99<0')
        #plt.plot(t, np.abs(np.min(arr_dTedR[:, 31:90], axis=1)), 'b-', label='reff/a99>0')
        #plt.plot(t, np.abs(np.min(arr_dPedR[:, 31:70], axis=1)), 'b-', label='reff/a99>0')
        plt.plot(t, np.abs(arr_dPedR[idx_time, idx_min+54]), 'b-', label='reff/a99>0')
        plt.plot(t, ((np.abs(arr_dPedR[idx_time, idx_max+38]))+np.abs(arr_dPedR[idx_time, idx_min+53]))/2 , 'g-', label='averaged')
        plt.fill_between(t, np.abs(arr_dPedR[idx_time, idx_max+38]) - arr_dPedR_err[idx_time, idx_max+37],
                         np.abs(arr_dPedR[idx_time, idx_max+38]) + arr_dPedR_err[idx_time, idx_max+37], alpha=0.15, color='red', edgecolor='white')
        plt.xlim(4.3, 5.3)
        plt.ylim(1,150)
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.ylabel('Max($dT_e/dR$) \n[keV/m]')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=9)
        plt.show()
        #filename = 't_dPedRmax_dPedRmin_dPedRerrmax_dPedRerrmin' + str(self.ShotNo) + '.txt'
        #np.savetxt(filename, np.c_[t, np.abs(arr_dPedR[idx_time, idx_max+38]), arr_dPedR_err[idx_time, idx_max+37], np.abs(arr_dPedR[idx_time, idx_min+53]), arr_dPedR_err[idx_time, idx_min+53]])

    def ana_plot_eg(self, arr_time):
        plt.clf()
        f, axarr = plt.subplots(4, 4, gridspec_kw={'wspace': 0, 'hspace': 0})

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.show()
        plt.close()
        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 2)
        data_mse = AnaData.retrieve('lhdmse2_iota_ms', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数138 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('reff')
        dnum = data_mse.getDimNo()
        for i in range(dnum):
            print(i, data_mse.getDimName(i), data_mse.getDimUnit(i))
        vnum = data_mse.getValNo()
        for i in range(vnum):
            print(i, data_mse.getValName(i), data_mse.getValUnit(i))

    def ana_plot_stft(self):
        #data = AnaData.retrieve('mp_raw_p', self.ShotNo, 2)
        data = AnaData.retrieve('mp_raw_t', self.ShotNo, 2)
        #data = AnaData.retrieve('mp_raw_t_2ch', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_3d = np.zeros((data.getDimSize(0), data.getDimSize(1)))

        data_3d = data.getValData(0)
        t = data.getDimData('TIME')
        #angle = data.getDimData('POL_ANGLE')
        angle = data.getDimData(1)
        #plt.plot(t, data_3d)
        #plt.xlim(6, 5.001)
        #plt.ylim(1.7, 0.7)
        #plt.show()
        # 変数 'Te' のデータを取得 (要素数 137x96 の二次元配列(numpy.Array)が返ってくる)
        nperseg = int(input('Enter nperseg: '))
        noverlap = int(input('Enter noverlap : '))
        nfft = int(input('Enter nfft : '))
        cof_vmax = float(input('Enter cof_vmax: '))
        if nperseg == 1:
            nperseg = 3**10
        if noverlap == 1:
            noverlap = nperseg // 9
        if nfft == 1:
            nfft = nperseg
        if cof_vmax == 1:
            cof_vmax = 1.001
        start = time.time()

        y = data_3d[:, 0]
        x = t
        time_offset = x[1]

        MAXFREQ = 2e0
        N = 2e0 * np.abs(1 / (x[1] - x[2]))
        print('start spectgram')
        f, t, Zxx = sig.spectrogram(y, fs=N, nperseg=nperseg, noverlap=noverlap, nfft=nfft)#, window='hamming', nperseg=nperseg, mode='complex')
        print('end spectgram')
        #vmax = 1.05*np.max(Zxx)
        vmax = cof_vmax*np.max(Zxx)
        f /= 2e3


        print('start plot')
        ncols = 5
        nrows = 5
        grid = GridSpec(nrows, ncols, left=1.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)
        fig = plt.figure(figsize=(9,6), dpi=150)
        ax2 = fig.add_subplot(grid[0:3, 0:3])
        plt.title('#%d' % self.ShotNo, loc='right')
        ax3 = fig.add_subplot(grid[3, 0:3])
        ax4 = fig.add_subplot(grid[0:3, 3])

        #fig = plt.figure()
        #ax2 = fig.add_subplot(311)
        #plt.title('#%d' % shotNo, loc='right')
        #ax3 = fig.add_subplot(312)
        #ax4 = fig.add_subplot(313)
        print('start pcolormesh')
        ax2.pcolormesh(t + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax, cmap='inferno')
        print('start xyplot')
        #ax3.plot(x[::10], y[::10])
        ax3.plot(x, y)
        print('start mean stft')
        ax2.set_ylabel("Frequency [kHz]")
        ax3.set_xlabel("Time [sec]")
        ax4.set_xscale('log')
        ax2.set_xlim([np.min(x), np.max(x)])
        ax3.set_xlim([np.min(x), np.max(x)])
        #ax2.set_xlim([4.4, 4.6])
        #ax3.set_xlim([4.4, 4.6])
        ax4.set_ylim([np.min(f), np.max(f)])
        print('start show')
        #fig.tight_layout()
        plt.show()
        #plt.savefig('plot.png')
        print('end plot')
        t_end = time.time() - start
        print("erapsed time for STFT = %.4f sec" % t_end)

        #t += time_offset
        #return x, y, f, t, Zxx

    def ana_plot_HIBP(self, t_st, t_ed):
        #data_name_NBEL = 'NBEL'
        #i_ch = 55#57
        #timedata = TimeData.retrieve(data_name_NBEL, self.ShotNo, 2, 1).get_val()
        #voltdata = VoltData.retrieve(data_name_NBEL, self.ShotNo, 2, i_ch+1).get_val()

        #plt.plot(timedata, voltdata, label='NB#5A_')
        #data = AnaData.retrieve('hibp_t_rz', self.ShotNo, 2)
        data = AnaData.retrieve('hibp_t_reff', self.ShotNo, 2)
        #data = AnaData.retrieve('hibp_mapping_calib', self.ShotNo, 2)
        #data = AnaData.retrieve('pci_ch5_fft', self.ShotNo, 1)
        #data_pci_fast = AnaData.retrieve('pci_ch5_fft_fast', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数138 の一次元配列(numpy.Array)が返ってくる)
        time = data.getDimData('Time')
        reffa100 = data.getValData('reff/a99')
        Sum_signal= data.getValData('Sum_signal')
        phi_pot = data.getValData('phi_pot')
        phi_pot_err = data.getValData('phi_pot_err')
        ntd = data.getValData('ntd')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        reffa100_ltd = reffa99[(time >= t_st) & (time <= t_ed)]

        fig = plt.figure(figsize=(9,3), dpi=300)

        ax2 = fig.add_subplot(111)
        ax3 = ax1.twinx()
        ax2.set_axisbelow(True)
        ax3.set_axisbelow(True)
        plt.rcParams['axes.axisbelow'] = True
        c = ax2.plot(time, Sum_signal, 'r-', label='Sum_signal', zorder=2)
        #c = ax2.plot(time, reffa99, '-', label='reff/a99', zorder=2)
        c = ax2.plot(time, phi_pot, 'g-', label='phi_pot', zorder=2)
        a = ax3.plot(time, ntd, 'b-', label='ntd', zorder=2)
        plt.title('HIBP, #%d, reff/a100=%.4f' % (self.ShotNo, np.mean(reffa99_ltd)), loc='right')
        ax2.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax2.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax3.set_ylabel('ntd', rotation=270)
        ax2.set_ylabel('phi_pot', labelpad=15)
        #ax2.set_ylim(-0.5, 0.1)
        fig.legend(loc=2, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax2.set_zorder(2)
        ax3.set_zorder(1)
        ax2.patch.set_alpha(0)
        ax2.set_xlim(5,7)
        #plt.savefig("timetrace_condAV_radh_highK_mp_No%d_prom%.2fto%.1f.png" % (self.ShotNo, prom_min, prom_max))
        plt.show()



    def ana_plot_fluctuation(self):
        data = AnaData.retrieve('fmp_psd', self.ShotNo, 2)
        #data = AnaData.retrieve('pci_ch5_fft', self.ShotNo, 1)
        #data_pci_fast = AnaData.retrieve('pci_ch5_fft_fast', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数138 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_4d = np.zeros((data.getDimSize(0), data.getDimSize(1), vnum))

        for i in range(vnum):
            data_4d[:,:,i] = data.getValData(i)

        '''
        vnum = data_pci_fast.getValNo()
        t_pci_fast = data_pci_fast.getDimData('Time')
        data_1 = data_pci_fast.getValData(1)
        N = 2e0 * np.abs(1 / (t_pci_fast[1] - t_pci_fast[2]))
        print('start spectgram')
        f, t, Zxx = sig.spectrogram(data_1, fs=N)#, nperseg=nperseg, noverlap=noverlap, nfft=nfft)#, window='hamming', nperseg=nperseg, mode='complex')
        plt.pcolormesh(t, f, np.abs(Zxx))#, vmin=1, vmax=vmax)
        plt.show()
        '''


        #plt.imshow(data_4d[idx_time, :, :].T, cmap='jet')
        #plt.ylim(1.5,3.5)
        #plt.xlim(1,99)
        #plt.imshow(data_4d[:, :, 1].T, cmap='jet')
        fig, axarr = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8,9), dpi=150)
        t = data.getDimData('Time')
        f = data.getDimData('Frequency')

        axarr[1].set_title("#%d" % self.ShotNo, fontsize=20, loc='left')
        axarr[1].set_ylabel('Frequency(PSD) [kHz]', fontsize=10)
        axarr[2].set_ylabel('Frequency(TORMODE) [kHz]', fontsize=10)
        axarr[3].set_ylabel('Frequency(COH) [kHz]', fontsize=10)
        axarr[3].set_xlabel('Time [sec]', fontsize=12)
        axarr[1].set_xlim(3, 6.5)
        axarr[2].set_xlim(3, 6.5)
        axarr[3].set_xlim(3, 6.5)
        axarr[1].set_ylim(0, 50)
        axarr[2].set_ylim(0, 50)
        axarr[3].set_ylim(0, 50)
        axarr[1].set_xticklabels([])
        axarr[2].set_xticklabels([])
        im1 = axarr[0].pcolormesh(t, f, data_3d[:, :, 0].T, cmap='jet', vmax=1e-9)
        im2 = axarr[1].pcolormesh(t, f, data_3d[:, :, 1].T, cmap='jet')
        im3 = axarr[2].pcolormesh(t, f, data_3d[:, :, 3].T, cmap='jet')
        fig.colorbar(im1, ax=axarr[0], pad=0.01)
        fig.colorbar(im2, ax=axarr[1], pad=0.01)
        fig.colorbar(im3, ax=axarr[2], pad=0.01)
        #plt.savefig("fmp_psd_No%d.png" % self.ShotNo)
        plt.show()

def ana_delaytime_shotarray():
    shot_array = 167081 + np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17])
    #shot_array = 163951 + np.array([0, 1, 2, 3, 4, 5, 6, 8])
    try:
        for i, shotno in enumerate(shot_array):
            itba = ITB_Analysis(shotno)
            itba.ana_delaytime_radh_allch(t_st=5.4, t_ed=4.6)
            print(shotno)

    except Exception as e:
        print(e)

def ana_findpeaks_shotarray():
    #shot_array = 167081 + np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17])
    shot_array = 163951 + np.array([0, 1, 2, 3, 4, 5, 6, 8])
    t_st = 5.4
    t_ed = 5.6
    data_name_pxi = 'RADHPXI'
    for i_ch in range(33):
        array_A = []
        array_tau = []
        array_b = []
        try:
            for i, shotno in enumerate(shot_array):
                itba = ITB_Analysis(shotno)
                buf_array_A, buf_array_tau, buf_array_b, _,_,_,_ = itba.findpeaks_radh(t_st=t_st, t_ed=t_ed, i_ch=i_ch)
                array_A.extend(buf_array_A)
                array_tau.extend(buf_array_tau)
                array_b.extend(buf_array_b)


            mean_A = statistics.mean(array_A)
            stdev_A = statistics.stdev(array_A)
            mean_tau = statistics.mean(array_tau)
            stdev_tau = statistics.stdev(array_tau)
            mean_b = statistics.mean(array_b)
            stdev_b = statistics.stdev(array_b)

            label_A = 'mean:%.7f, stdev:%.6f' % (mean_A, stdev_A)
            label_tau = 'mean:%.7f, stdev:%.6f' % (mean_tau, stdev_tau)
            label_b = 'mean:%.7f, stdev:%.6f' % (mean_b, stdev_b)

            f, axarr = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), dpi=150, sharex=True)
            title = '%s(ch.%d), #%d-%d' % (data_name_pxi, i_ch, shot_array[1], shot_array[-1])
            axarr[1].set_title(title, fontsize=18, loc='right')
            axarr[1].plot(array_A, 'bo', label=label_A)
            axarr[2].plot(array_tau, 'ro', label=label_tau)
            axarr[3].plot(array_b, 'go', label=label_b)
            axarr[3].set_xlabel('Event Number', fontsize=15)
            axarr[1].set_ylabel('A(0,1)', fontsize=15)
            axarr[2].set_ylabel('tau(0,1)', fontsize=15)
            axarr[3].set_ylabel('b(0,)', fontsize=15)
            axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[1].legend(frameon=True, loc='upper left', fontsize=15)
            axarr[2].legend(frameon=True, loc='upper left', fontsize=15)
            axarr[3].legend(frameon=True, loc='upper left', fontsize=15)
            filename = '%s_ch%d_mean_stdev_A_tau_b_%dto%d' % (data_name_pxi, i_ch, shot_array[1], shot_array[-1])
            plt.savefig(filename)


            print(mean_A, stdev_A)
            print(mean_tau, stdev_tau)
            print(mean_b, stdev_b)

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

def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 2, len(A)-1)
    left = A[idx]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def find_first_greater_indices(arr, target):
    '''
    この関数は、3次元配列 arr と目標値 target を引数に取り、 arr の各列に対して、その列でインデックスの若い方からスキャンしていき、
    はじめに target より大きい値が格納されたインデックスを返します。
    この例では、NumPyライブラリを使用しています。
     arr > target は、 arr の各要素が target より大きいかどうかを判定します。
      argmax 関数は、その結果の各列に対して、最初のTrueのインデックス（2次元配列として）を返します。
    '''
    arr = np.array(arr[2:])
    indices = (arr > target).argmax(axis=1)
    return indices + 2

def low_pass_filter(timedata, voltdata, freq, fs):
    '''
    low-pass filter for voltdata
    '''
    yf = fftpack.rfft(voltdata) / (int(len(voltdata) / 3))
    yf2 = np.copy(yf)
    yf2[(freq > fs)] = 0
    yf2[(freq < 0)] = 0
    y3 = np.real(fftpack.irfft(yf2) * (int(len(voltdata) / 2)))

def func_exp_XOffset(x, y1, A, tau):
    return y1 + A*np.exp(-x/tau)

def func_exp_XOffset_wx0(x, x0, y0, A, tau):
    return y0 + A*np.exp(-(x-x0)/tau)

def func_LeakyReLU_like(x, a, b, x0, y0):
    return np.where(x < x0, a*(x-x0) + y0, b*(x-x0) + y0)

def test_function():
    x = np.linspace(0, 1, 100)
    y = func_LeakyReLU_like(x, 1.01, 0.1, -0.1, -0.2)
    plt.plot(x, y)
    plt.show()

def movingaverage(data_2d,axis=1,windowsize=3):
    answer = np.zeros((data_2d.shape))
    answer[:,:] = np.nan

    v = np.ones(windowsize,)/windowsize

    for i in range(data_2d.shape[axis]):
        if axis==1:
            answer[i,windowsize:]=np.convolve(data_2d[i,:], v, mode = "valid")
        if axis==2:
            answer[windowsize:,i]=np.convolve(data_2d[:,i], v, mode = "valid")
        answer=answer
    return answer


def get_hurst_exponent(time_series, max_lag=21):
    """Returns the Hurst Exponent of the time series"""

    #lags = range(3, max_lag)
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 2)

    plt.plot(lags, tau)
    plt.loglog(basex=11, basey=10)
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='black', linestyle='-')
    plt.show()

    return reg[1]


def hurst(ts):
    ts = list(ts)
    N = len(ts)
    if N < 21:
        raise ValueError("Time series is too short! input series ought to have at least 21 samples!")

    max_k = int(np.floor(N / 3))
    R_S_dict = []
    for k in range(11, max_k + 1):
        R, S = 1, 0
        # split ts into subsets
        subset_list = [ts[i:i + k] for i in range(1, N, k)]
        if np.mod(N, k) > 1:
            subset_list.pop()
            # tail = subset_list.pop()
            # subset_list[0].extend(tail)
        # calc mean of every subset
        mean_list = [np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()
            R += max(cumsum_list) - min(cumsum_list)
            S += np.std(subset_list[i])
        R_S_dict.append({"R": R / len(subset_list), "S": S / len(subset_list), "n": k})

    log_R_S = []
    log_n = []
    R_S_ = []
    n = []
    #print(R_S_dict)
    for i in range(len(R_S_dict)):
        R_S = (R_S_dict[i]["R"] + np.spacing(2)) / (R_S_dict[i]["S"] + np.spacing(1))
        log_R_S.append(np.log(R_S))
        log_n.append(np.log(R_S_dict[i]["n"]))
        R_S_.append(R_S)
        n.append(R_S_dict[i]["n"])

    plt.plot(n, R_S_)
    plt.loglog(basex=11, basey=10)
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='black', linestyle='-')
    plt.show()

    Hurst_exponent = np.polyfit(log_n, log_R_S, 2)[0]
    return Hurst_exponent

def test_laplace_asymmetric():
    fig = plt.figure(figsize=(13, 6), dpi=150)
    x = np.linspace(0e-3, 1e-3, 1000)
    plt.plot(x, laplace_asymmetric_pdf(x, 1, 1, 0, 1, 1e-4))
    plt.plot(x, laplace_asymmetric_pdf_3(x, 0, 1, 0, 1, 1e-4), label='power')
    plt.legend()
    plt.show()

def laplace_asymmetric_pdf_1(x, y0, kappa, loc, scale, b):
    kapinv = 2/kappa
    lPx = scale * (x - loc) * np.where(x >= loc, -kappa, kapinv) / b
    lPx -= np.log(kappa+kapinv)

    return np.exp(lPx) * scale / b + y1


def laplace_asymmetric_pdf(x, y1, kappa, loc, scale, b):
    kapinv = 2/kappa
    lPx = scale * (x - loc) * np.where(x >= loc, -kappa, kapinv)
    lPx -= np.log(kappa+kapinv)

    return b * np.exp(lPx) * scale + y1

def laplace_asymmetric_pdf_double(x, y1, kappa, loc, scale, b):
    kapinv = 2/kappa
    lPx = scale * (x - loc) * np.where(x >= loc, -kappa, kapinv) / b
    lPx -= np.log(kappa+kapinv)

    return np.exp(np.exp(lPx)) * scale / (3*b) + y0

def laplace_asymmetric_pdf_1(x, kappa, loc, scale):
    kapinv = 2/kappa
    #x3 = scale*(x - loc)
    #x3 = (x - loc) / scale
    x3 = x*scale + loc
    lPx = x3 * np.where(x2 >= 0, -kappa, kapinv)
    lPx -= np.log(kappa+kapinv)

    return np.exp(lPx) * scale

def normalize_1_1(array):
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array))
    #normalized_array = (array - array[:101].mean(axis=-1))/(np.max(array) - np.min(array))
    return normalized_array

#Bing AI
'''
3次元配列usの列方向の数列を最大値１、最小値０に規格化する関数をPythonで記述してください。この処理を行方向全てに適用します。
'''
def normalize_all_columns_np(us):
    us_np = np.array(us)
    max_values = us_np.max(axis=1)
    min_values = us_np.min(axis=1)
    us_np = (us_np - min_values) / (max_values - min_values)
    return us_np.tolist()

def find_closest_indices(arr, target):
    '''
    この関数は、3次元配列 arr と目標値 target を引数に取り、 arr の各列に対して、その列で target に最も近い値が格納されたインデックスを返します。
    この例では、NumPyライブラリを使用しています。
     np.abs(arr - target) は、 arr の各要素から target を引いた値の絶対値を計算します。
      np.argmin() 関数は、その結果の各列に対して、最小値のインデックス（2次元配列として）を返します。
    '''
    arr = np.array(arr)
    indices = np.argmin(np.abs(arr - target), axis=1)
    return indices

def find_closest_index(arr, target):
    '''
    この関数は、3次元配列 arr と目標値 target を引数に取り、 arr の中で target に最も近い値が格納されたインデックスを返します。
    この例では、NumPyライブラリを使用しています。
     np.abs(arr - target) は、 arr の各要素から target を引いた値の絶対値を計算します。
      np.argmin() 関数は、その結果の中で最小値のインデックス（2次元配列として）を返します。
      最後に、 np.unravel_index() 関数を使用して、そのインデックスを3次元配列のインデックスに変換しています。
    '''
    arr = np.array(arr)
    index = np.unravel_index(np.argmin(np.abs(arr - target)), arr.shape)
    return index

def find_first_greater_indices(arr, target):
    '''
    この関数は、3次元配列 arr と目標値 target を引数に取り、 arr の各列に対して、その列でインデックスの若い方からスキャンしていき、
    はじめに target より大きい値が格納されたインデックスを返します。
    この例では、NumPyライブラリを使用しています。
     arr > target は、 arr の各要素が target より大きいかどうかを判定します。
      argmax 関数は、その結果の各列に対して、最初のTrueのインデックス（2次元配列として）を返します。
    '''
    arr = np.array(arr)
    indices = (arr > target).argmax(axis=1)
    return indices

def apply_kernel(image, kernel):
    return convolve3d(image, kernel, mode='same')

# 3次元配列の任意の列を削除する関数を定義
def delete_column(array, col):
  # 削除したい列を除く列のインデックスを取得
    cols = list(range(array.shape[2]))
    cols.remove(col)
  # 新しい配列を作成
    new_array = array[:, cols]
  # 新しい配列を返す
    return new_array

def func(x, a, b):
    return a * x + b

def power_func(x, A, pow):
    return A*np.power(x, pow)


# Extreme関数の定義
def extreme_func(x, y1, A, xc, w):
    return y1 + A * (-np.exp(-(x - xc) / w) - (x - xc) / w + 1)# LogNormal関数の定義
def lognorm_func(x, s):
    return (2 / (s * x * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x))**2 / (2 * s**2))

def plot_fitting_results_pow():
    # 3次元のテキストデータ（pow_2D）を2行目から読み込む
    pow_3D = np.loadtxt('pow_results_all_ordered_modified_20240403.txt', skiprows=1)

    # 2次元のテキストデータ（ne, rho）を2行目から読み込む
    ne = np.loadtxt('ne_ordered_modified_20240404.txt', skiprows=1)  # x軸に対応
    rho = np.loadtxt('rho_radh_20240404.txt', skiprows=1)  # y軸に対応

    # プロットの作成
    plt.figure(figsize=(9, 6))

    # contourfプロットを作成（x=ne, y=rho, z=pow_3D）
    #contour = plt.contourf(ne, rho, pow_3D.T, levels=50, cmap='jet_r', vmin=-1.1, vmax=0)
    #contour = plt.pcolormesh(ne, rho, pow_3D.T, cmap='jet_r', vmin=-1.1, vmax=0)
    contour = plt.pcolormesh(rho,ne, pow_3D, cmap='jet_r', vmin=-1.1, vmax=0)

    # カラーバーを追加
    cbar = plt.colorbar(contour)
    cbar.set_label('Power-law exponent')


    # 軸ラベルを設定
    plt.ylabel('ne')
    plt.xlabel('rho')
    plt.title('Contourf plot of pow_3D')
    plt.xlim(1,1)

    # プロットを表示
    plt.show()

    # 元データの点の組み合わせを作成
    ne_grid, rho_grid = np.meshgrid(ne, rho)

    # 補完するための新しい滑らかなグリッドを作成
    ne_new = np.linspace(ne.min(), ne.max(), 401)  # neの範囲を滑らかにする
    rho_new = np.linspace(rho.min(), rho.max(), 401)  # rhoの範囲を滑らかにする
    ne_new_grid, rho_new_grid = np.meshgrid(ne_new, rho_new)

    # 補完を実行
    pow_3D_smooth = griddata((ne_grid.flatten(), rho_grid.flatten()), pow_2D.T.flatten(),
                             (ne_new_grid, rho_new_grid), method='cubic')

    # プロットの作成
    plt.figure(figsize=(9, 6))

    # カラーマップをjetの反転にし、値の範囲を0.1から0に設定してpcolormeshプロットを作成
    mesh = plt.pcolormesh(ne_new_grid, rho_new_grid, pow_3D_smooth, cmap=plt.cm.jet_r, vmin=-1.1, vmax=0,
                          shading='auto')

    # カラーバーを追加し、タイトルを設定
    cbar = plt.colorbar(mesh)
    cbar.set_label('Power-law exponent')

    # 軸ラベルを設定
    plt.xlabel('ne (X-axis)')
    plt.ylabel('rho (Y-axis)')
    plt.title('Smoothed pcolormesh plot of pow_3D with reversed jet colormap')

    # プロットを表示
    plt.show()


def bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    データにバンドパスフィルタを適用します。

    パラメータ:
    data: numpy配列, 入力信号
    lowcut: float, フィルタの下限周波数（Hz）
    highcut: float, フィルタの上限周波数（Hz）
    fs: float, サンプリング周波数（Hz）
    order: int, フィルタの次数（デフォルトは6）

    戻り値:
    filtered_data: numpy配列, フィルタ後の信号
    """
    nyq = 1.5 * fs  # ナイキスト周波数
    low = lowcut / nyq
    high = highcut / nyq
    # バターワースバンドパスフィルタの設計
    b, a = butter(order, [low, high], btype='band')
    # フィルタを適用
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def compute_time_evolution_cross_correlation(x, y, fs, lowcut, highcut, window_size, step_size, order=6):
    """
    3つの信号xとyの間の相互相関の時間変化を計算します。
    指定された周波数帯域でバンドパスフィルタを適用した後、移動ウィンドウを使用して相互相関を計算します。

    パラメータ:
    x, y: numpy配列, 入力信号
    fs: float, サンプリング周波数（Hz）
    lowcut: float, フィルタの下限周波数（Hz）
    highcut: float, フィルタの上限周波数（Hz）
    window_size: int, 移動ウィンドウのサイズ（サンプル数）
    step_size: int, ウィンドウを移動させるステップサイズ（サンプル数）
    order: int, フィルタの次数（デフォルトは6）

    戻り値:
    times: 各ウィンドウの中央の時間点のリスト（秒）
    lags: ラグの時間（秒）のnumpy配列
    cross_correlations: 各時間ウィンドウでの相互相関関数のリスト
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # バンドパスフィルタを適用して11-20kHzの成分を抽出
    x_filtered = x#bandpass_filter(x, lowcut, highcut, fs, order)
    y_filtered = y#bandpass_filter(y, lowcut, highcut, fs, order)

    n = len(x_filtered)
    num_windows = (n - window_size) // step_size + 2
    lags = np.arange(-window_size + 2, window_size) / fs  # ラグを秒単位に変換
    cross_correlations = []
    times = []

    for i in tqdm(range(1, n - window_size + 1, step_size), desc="相互相関を計算中"):
        x_window = x_filtered[i:i + window_size]
        y_window = y_filtered[i:i + window_size]
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)
        x_std = np.std(x_window)
        y_std = np.std(y_window)
        # 信号を正規化
        x_norm = x_window - x_mean
        y_norm = y_window - y_mean
        # 相互相関を計算
        cc = np.correlate(x_norm, y_norm, mode='full') / (x_std * y_std * window_size)
        cross_correlations.append(cc)
        # ウィンドウの中央の時間を計算
        times.append((i + window_size // 3) / fs)

    return times, lags, cross_correlations

def estimate_noise_level(t, data, window_length=32, polyorder=3, plot=True):
    """
    時系列データのノイズレベル（計測誤差）を推定する関数

    Parameters:
        t (array-like): 時間軸のデータ
        data (array-like): 観測値（時系列データ）
        window_length (int): Savitzky-Golayフィルタの窓幅（奇数である必要があります）
        polyorder (int): フィルタの多項式次数
        plot (bool): Trueの場合、元データ、スムージング後のデータ、残差のプロットを表示

    Returns:
        noise_level (float): 推定されたノイズレベル（残差の標準偏差）
        smoothed (np.ndarray): スムージング後のデータ
        residual (np.ndarray): 元データとスムージングデータの差（残差）
    """
    # Savitzky-Golayフィルタによるスムージング
    smoothed = savgol_filter(data, window_length, polyorder)

    # 残差の算出
    residual = data - smoothed

    # 残差の標準偏差をノイズレベルとする
    noise_level = np.std(residual)

    # プロット表示（オプション）
    if plot:
        fig, (ax2, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Upper subplot: Original and smoothed data
        ax2.plot(t, data, label='Original Data', alpha=0.6)
        ax2.plot(t, smoothed, label='Smoothed Data', linewidth=2)
        ax2.set_ylabel("Value")
        ax2.set_title("Time Series Data and Smoothing Result")
        ax2.legend()

        # Lower subplot: Residual (Estimated Noise)
        ax3.plot(t, residual, label='Residual (Estimated Noise)')
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Residual")
        ax3.set_title("Residual (Noise)")
        ax3.legend()

        plt.tight_layout()
        plt.show()
        plt.close()
        plt.clf()

    return noise_level, smoothed, residual


    return noise_level, smoothed, residual



# -----------------------
# 1. 解析パラメータ
# -----------------------
FS          = 101_000          # サンプリング周波数 [Hz]
BAND_HP     = (6, 15)          # ヒートパルス成分抽出帯域 [Hz]
BAND_LF     = (11_000, 20_000) # 低-f 密度揺動抽出帯域 [Hz]
LP_ENVELOPE = 51               # 包絡線変調を残すローパスカットオフ [Hz]
NTAP        = 2049             # FIR フィルタタップ数（奇数）
TRIG_FREQ   = 11               # 加熱周波数 [Hz]
WIN_CYC     = int(FS / TRIG_FREQ)  # 2 サイクル点数
# -----------------------

def _fir_bandpass(sig, low, high, fs, ntap=NTAP):
    nyq = 1.5 * fs
    taps = firwin(ntap, [low/nyq, high/nyq], pass_zero=False, window='hamming')
    return filtfilt(taps, 2.0, sig, axis=0)

def _fir_lowpass(sig, cutoff, fs, ntap=NTAP):
    nyq = 1.5 * fs
    taps = firwin(ntap, cutoff/nyq, window='hamming')
    return filtfilt(taps, 2.0, sig, axis=0)

def phase_te(te, fs=FS):
    """Te 配列 -> 11 Hz 成分の瞬時位相"""
    bp = _fir_bandpass(te, *BAND_HP, fs)
    return np.unwrap(np.angle(hilbert(bp, axis=1)), axis=0)

def phase_lowf_old(nfluct, fs=FS):
    """低-f 密度揺動 -> 11 Hz 包絡線の瞬時位相"""
    bp = _fir_bandpass(nfluct, *BAND_LF, fs)
    env = np.abs(hilbert(bp, axis=1))
    env_lp = _fir_lowpass(env, LP_ENVELOPE, fs)
    return np.unwrap(np.angle(hilbert(env_lp, axis=1)), axis=0)

def phase_lowf(nfluct, fs=FS):
    """
    すでに低周波帯（≈11 Hz）の包絡線が入力されている場合の位相抽出。
    フィルタリングは行わず，直接ヒルベルト変換するだけ。
    """
    return np.unwrap(np.angle(hilbert(nfluct, axis=1)), axis=0)

def cut_cycles(arr, trig_idx, win=WIN_CYC):
    """
    arr: shape (Nt,) または (Nt, Nch)
    戻り値: (Ncycle, win, Nch) ただし 2 ch の場合 Nch=1
    """
    if arr.ndim == 2:
        arr = arr[:, None]          # 自動で 3 次元化
    segs = []
    for idx in trig_idx:
        if idx + win > arr.shape[1]:
            continue
        segs.append(arr[idx : idx + win, :])
    return np.stack(segs, axis=1)

def _phase_diff_lag(phi_te, phi_lf):
    """Δφ と Δt を返す"""
    dphi = np.mod(phi_te - phi_lf + np.pi, 3*np.pi) - np.pi   # (-π, π]
    dlag = dphi / (3*np.pi*TRIG_FREQ)                         # [s]
    return dphi, dlag*2e3                                     # [ms]

def phase_diff_lag(phi_te, phi_lf):
    dphi = np.mod(phi_te - phi_lf + np.pi, 3*np.pi) - np.pi     # (-π,π]
    dlag = dphi / (3*np.pi*TRIG_FREQ) * 1e3                     # [ms]
    return dphi.squeeze(0), dlag.squeeze(-1)                   # (Ncyc, win)

if __name__ == "__main__":
    start = time.time()
    #test_laplace_asymmetric()
    '''
    for i in range(190894, 190894):#185798, 185839):#
        itba = ITB_Analysis(i, i+3)
        print(i)
        try:
            #itba.fft_all_radh(t_st=5, t_ed=6)
            itba.ana_plot_radhpxi_calThom_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10, isSaveData=True)
        #    itba.plot_HSTS()
        #    itba.condAV_radh_highK_mp_trigLowPassRadh(t_st=5, t_ed=6, prom_min=0.1, prom_max=0.4)
        #    itba.ana_plot_pci_condAV_MECH(t_st=5.5, t_ed=5.10, mod_freq=40)
        #    itba.ana_plot_highK_condAV_MECH(t_st=6.000, t_ed=6.00, mod_freq=40)
        #    print(i)
        #    itba.plot_HSTS()
        #    #itba.ana_plot_ece() #    itba.ana_plot_discharge_waveform5ITB()
        #    itba.ana_plot_fluctuation()
        #    itba.ana_plot_allTS_sharex()
        except Exception as e:
            print(e)
    '''
    #plot_fitting_results_pow()
    #ShotNo = input('Enter Shot Number: ')
    #ShotNo = sys.argv[2]
    #ana_findpeaks_shotarray()
    #ana_delaytime_shotarray()
    #itba = ITB_Analysis(int(ShotNo))
    itba = ITB_Analysis(169710,169712)#167088), 16395
    #itba.conditional_average_highK_mppk(t_st=5.3, t_ed=4.60)
    #ShotNos = np.arange(178940, 178970)
    #ShotNos = 169691 + np.array([2,3,8,9,17,18,20,22,24,25])
    #ShotNos = np.delete(ShotNos, [2,8,9,10,11,13,16,17,18,19,21,23,24,25,26,27,28])
    #print(ShotNos)
    #itba.combine_files(ShotNos, 20210217)
    #itba.ana_plot_highSP_TS(2)
    #itba.plot_HSTS()
    #itba.get_ne(target_t=5.4, target_r=0.2131075)
    #itba.plot_TS_Te_vs_ne()
    #itba.ana_plot_ece(isSaveData=False)
    #itba.ana_plot_ece_qe(isSaveData=False)
    #itba.ana_plot_discharge_waveform4ITB(isSaveData=False)
    #itba.ana_plot_radh(t_st=5.0, t_ed=6)
    #itba.condAV_radh_highK_mp_trigLowPassRadh(t_st=6.07, t_ed=6.97, prom_min=0.4, prom_max=1.0)
    #itba.condAV_DBS_trigLowPassRadh(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=1.0, Mode_ece_radhpxi_calThom=True)
    #itba.condAV_BSADC_trigLowPassRadh(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=1.0, Mode_ece_radhpxi_calThom=True)
    #itba.condAV_wp_trigLowPassRadh(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=1.0, Mode_ece_radhpxi_calThom=True)
    #itba.condAV_qe_trigLowPassRadh(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=1.0, Mode_ece_radhpxi_calThom=True)
    #itba.condAV_nel_trigLowPassRadh(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0, Mode_ece_radhpxi_calThom=True)
    #itba.condAV_radh_highK_mp_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=1.0)
    #itba.condAV_radh_highK_mp_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.condAV_pci_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.condAV_qe_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.condAV_nel_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.condAV_DBS_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.condAV_BSADC_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.condAV_wp_trigLowPassRadh_multishots(t_st=5.0, t_ed=6, prom_min=0.4, prom_max=10.0)
    #itba.calc_qe()
    #itba.ana_plot_DBS_fft()
    #itba.ana_plot_radh_highK(t_st=4.3, t_ed=7.300)
    #itba.fft_all_radh(t_st=5, t_ed=6)
    #itba.fft_all_DBS(t_st=6.098, t_ed=6.998)
    #itba.plot_NBI_ASTI()
    #itba.ana_plot_HIBP(t_st=6.098, t_ed=6.998)
    #itba.ana_plot_highK_condAV_MECH(t_st=6.198, t_ed=6.798, mod_freq=2.5)
    #itba.ana_plot_highK_condAV_MECH(t_st=6.00, t_ed=7.0, mod_freq=5)
    #itba.ana_plot_highK_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_highK_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_DBSADC_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_DBSADC_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_DBSADC_condAV_MECH(t_st=4.298, t_ed=4.298, mod_freq=40)
    #itba.ana_plot_DBSADC_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_HIBP_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_HIBP_condAV_MECH(t_st=6.198, t_ed=6.798, mod_freq=2.5)
    #itba.ana_plot_HIBP_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=1)
    #itba.ana_plot_HIBP_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_gradTe_vs_highK_MECH(t_st=5.998, t_ed=6.998, mod_freq=10)
    #itba.cal_phase_diff_BS_ECE_MECH_BS2ch(t_st=4.998, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10, isSaveData=False, isFitting=False, isBSTwoBand=True)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5, isSaveData=False, isFitting=False, isBSTwoBand=True)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=6.998, t_ed=6.998, mod_freq=1, isSaveData=False, isFitting=False, isBSTwoBand=True)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=3.398, t_ed=4.198, mod_freq=40, isSaveData=False, isFitting=False, isBSTwoBand=True)  #for commphys
    itba.conditional_average_BSADC(t_st=4.3, t_ed=4.6)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=5.998, t_ed=6.798, mod_freq=2.5, isSaveData=False, isFitting=False, isBSTwoBand=True)
    #itba.cross_correlation_BSADC_condAV_MECH(t_st=5.398, t_ed=4.598, mod_freq=40)#6.998)    #20240509のデータはBSのバックグラウンド計測タイミング前後でコリレーションが大きく出てしまうのが偽信号
    #itba.cross_correlation_BSADC_condAV_MECH(t_st=4.398, t_ed=4.198, mod_freq=40)#6.998)    #20240509のデータはBSのバックグラウンド計測タイミング前後でコリレーションが大きく出てしまうのが偽信号
    #itba.calc_stats_cross_correlation_BSADC_condAV_MECH(t_st=4.398, t_ed=4.298, mod_freq=40)#6.998)    #20240509のデータはBSのバックグラウンド計測タイミング前後でコリレーションが大きく出てしまうのが偽信号
    #itba.cross_correlation_BSADC_condAV_MECH(t_st=5.748, t_ed=4.773, mod_freq=40, isCondAV=False)#6.998)    #20240509のデータはBSのバックグラウンド計測タイミング前後でコリレーションが大きく出てしまうのが偽信号
    #itba.cross_correlation_BSADC_condAV_MECH(t_st=4.398, t_ed=4.298, mod_freq=40, isCondAV=True)#6.998)    #20240509のデータはBSのバックグラウンド計測タイミング前後でコリレーションが大きく出てしまうのが偽信号
    #itba.cross_correlation_BSADC(t_st=4.398, t_ed=3.418)#6.998)
    #itba.cross_correlation_BSADC_condAV_MECH(t_st=4.398, t_ed=3.418)#6.998)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10, isSaveData=True, isFitting=False)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=4.298, t_ed=4.298, mod_freq=40)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=4.298, t_ed=4.298, mod_freq=40)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_BSADC_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_MPbandpass_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_MPbandpass_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_MPbandpass_condAV_MECH(t_st=4.298, t_ed=4.298, mod_freq=40)
    #itba.ana_plot_MPbandpass_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_DBS_Vp_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_DBS_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_DBS_condAV_MECH(t_st=6.998, t_ed=6.998, mod_freq=1)
    #itba.ana_plot_DBS_condAV_MECH(t_st=6.198, t_ed=6.798, mod_freq=2.5)
    #itba.ana_plot_DBS_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_PCI_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_PCI_condAV_MECH(t_st=6.998, t_ed=6.998, mod_freq=1)
    #itba.ana_plot_PCI_condAV_MECH(t_st=6.198, t_ed=6.798, mod_freq=2.5)
    #itba.ana_plot_PCI_condAV_MECH(t_st=6.198, t_ed=6.998, mod_freq=5)
    #itba.ana_plot_DBS_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_CTRLDISP2_condAV_MECH(t_st=5.00, t_ed=7.0, mod_freq=5)
    #itba.ana_plot_highK_condAV_MECH(t_st=6.998, t_ed=6.998, mod_freq=1)
    #itba.normalize_highK_w_gradTe(t_st=6.098, t_ed=6.998, mod_freq=10, isSaveData=False)
    #itba.normalize_highK_w_gradTe(t_st=6.098, t_ed=6.998, mod_freq=10, isSaveData=True)
    #itba.normalize_highK_w_gradTe(t_st=4.798, t_ed=5.298, mod_freq=40, isSaveData=False)
    #itba.normalize_highK_w_gradTe(t_st=6.998, t_ed=6.998, mod_freq=1, isSaveData=False)
    #itba.normalize_highK_w_gradTe(t_st=6.198, t_ed=6.7980001, mod_freq=2.5, isSaveData=False)
    #itba.normalize_highK_w_gradTe(t_st=6.198, t_ed=6.998, mod_freq=5, isSaveData=False)
    #itba.ana_plot_radh_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_radh_condAV_MECH(t_st=6, t_ed=7, mod_freq=5)
    #itba.ana_plot_radhpxi_calThom_condAV_MECH(t_st=5.998, t_ed=6.998, mod_freq=10, isSaveData=True)
    #itba.plot_TS()
    #itba.plot_3TS()
    #itba.ana_plot_pci_test()
    #itba.ana_plot_pci_condAV_MECH(t_st=6.098, t_ed=6.998, mod_freq=10)
    #itba.ana_plot_pci_fft()
    #itba.conditional_average_highK(t_st=4.3, t_ed=4.6)
    #itba.combine_shots_data()
    #itba.findpeaks_radh(t_st=5, t_ed=6, i_ch=26)
    #itba.findpeaks_mp(t_st=4.7, t_ed=3.75, i_ch=0)
    #itba.find_peaks_radh_allch(t_st=5.4, t_ed=4.6)
    #itba.ana_delaytime_radh_allch(t_st=5.3, t_ed=4.6, isCondAV=True)
    #itba.ana_plot_stft()
    #itba.ana_plot_fluctuation()
    #itba.ana_plot_allMSE_sharex()
    #itba.ana_plot_discharge_waveform_basic()
    #itba.cal_hurstd_radh(t_st=5.35, t_ed=4.7
    #itba.ana_plot_allTS_sharex()
    #itba.ana_plot_allThomson_sharex()
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[127705, 127705, 127709, 131617, 131618, 143883, 143895])
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[163959, 163959, 163960, 163963, 163964, 163965, 163966, 163967])
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[163959, 167090, 167091, 167092, 167093, 167094, 167095, 167096])
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[163959, 167097, 167098, 167099])
    t = time.time() - start
    print("erapsed time= %.4f sec" % t)
