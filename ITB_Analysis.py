import sys
from nifs.anadata import *
from nifs.voltdata import *
from nifs.timedata import *
from matplotlib.gridspec import GridSpec
from ftplib import FTP
from scipy import fftpack
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import signal
import scipy.signal as sig
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import statistics
from matplotlib.colors import LogNorm
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
            axarr[0,idx].set_ylim(0, 2.5)
            axarr[1,idx].set_ylim(0, 15.5)
            title_time = ('t=%.3fsec') % (t[idx+90]/1000)
            axarr[0,idx].set_title(title_time)
            axarr[1, idx].set_xlabel('R [m]')


        axarr[0, 0].set_ylabel(r'$n_e [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e [keV]$')


        plt.savefig("Thomson_Te_ne_No%d.png" % self.ShotNo)
        #plt.show()
        #plt.close()

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
        r_unit = data.getDimUnit('m')

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

    def ana_plot_highSP_TS(self, arr_time):
        plt.clf()
        f, axarr = plt.subplots(3, 4, gridspec_kw={'wspace': 0, 'hspace': 0})

        for i, ax in enumerate(f.axes):
            ax.grid('on', linestyle='--')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        data = AnaData.retrieve('high_time_res_thomson', self.ShotNo, 1)

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
        arr_tau_A_ltd = []
        arr_err_timedata_peaks_ltd = []
        if isCondAV:
            print("timedata and voltdata_array are loaded.")
            timedata, voltdata_array = self.combine_shots_data()
        for i_ch in range(32):
            try:
                print(self.ShotNo, i_ch)
                if isCondAV:
                    _,array_tau_A_ltd,_,timedata_peaks_ltd,err_timedata_peaks_ltd = self.findpeaks_radh(t_st, t_ed, i_ch, timedata=timedata+1, voltdata_array=voltdata_array, isCondAV=True)
                    timedata_peaks_ltd -= 1
                else:
                    _,array_tau_A_ltd,_,timedata_peaks_ltd,err_timedata_peaks_ltd = self.findpeaks_radh(t_st, t_ed, i_ch)
                ydata = r[i_ch]*np.ones(len(timedata_peaks_ltd))
                #plt.plot(timedata_peaks_ltd, ydata, 's')
                #plt.plot(ydata, timedata_peaks_ltd, 'rs')
                plt.plot(ydata, array_tau_A_ltd, 'rs')
                arr_ydata.extend(ydata)
                arr_timedata_peaks_ltd.extend(timedata_peaks_ltd)
                arr_err_timedata_peaks_ltd.extend(err_timedata_peaks_ltd)
                arr_tau_A_ltd.extend(array_tau_A_ltd)
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
        #filename = 'peaktime_%s_raw_tau_A_%d' % (data_name_pxi, self.ShotNo)
        filename = 'risetime_%s_raw_tau_A_%d_condAV' % (data_name_pxi, self.ShotNo)
        #plt.savefig(filename)
        #filename = 'risetime_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        #np.savetxt(filename, np.r_[arr_ydata, arr_timedata_peaks_ltd, arr_err_timedata_peaks_ltd])
        #filename = 'time_tau_errtau_radh_condAV_SN' + str(self.ShotNo) + '.txt'
        #np.savetxt(filename, np.r_[arr_ydata, arr_tau_A_ltd, arr_err_timedata_peaks_ltd])
        plt.show()


    def find_peaks_radh_allch(self, t_st, t_ed):
        for i_ch in range(32):
            try:
                self.findpeaks_radh(t_st, t_ed, i_ch)
            except Exception as e:
                print(e)


    def findpeaks_radh(self, t_st, t_ed, i_ch, timedata=None, voltdata_array=None, isCondAV=None):
        if isCondAV:
            print("timedata and voltdata_array are loaded.")
            timedata_ltd = timedata

        else:
            data_name_info = 'radhinfo' #radlinfo
            data_name_pxi = 'RADHPXI' #RADLPXI
            data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

            # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
            r = data_radhinfo.getValData('R')
            timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
            idx_time_st = find_closest(timedata, t_st)
            idx_time_ed = find_closest(timedata, t_ed)
            voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
            voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i_ch+1).get_val()
            voltdata_array[:, i_ch] = voltdata[idx_time_st:idx_time_ed]
            timedata_ltd = timedata[idx_time_st:idx_time_ed]
            #plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
            label = 'R=%.3fm' % r[i_ch]
        peaks, _ = find_peaks(voltdata_array[:, i_ch], prominence=0.001, width=40)
        #peaks, _ = find_peaks(x, distance=5000)

        '''
        f, axarr = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8,8), dpi=150, sharex=True)
        data_name_pxi = 'RADHPXI'  # RADLPXI

        axarr[0].set_xlim(t_st, t_ed)
        axarr[0].plot(timedata_ltd, voltdata_array[:, i_ch])
        axarr[0].plot(timedata_ltd[peaks], voltdata_array[peaks, i_ch], "or")
        title = '%s(ch.%d), #%d' % (data_name_pxi, i_ch, self.ShotNo)
        axarr[0].set_title(title, fontsize=18, loc='right')
        '''

        array_A = []
        array_tau = []
        array_err_tau = []
        array_tau_ltd = []
        array_err_tau_ltd = []
        array_b = []
        array_x0 = []
        array_err_x0 = []

        for i_peak in range(len(peaks)):
            width = 5000
            #width_2 = 2000
            #pre_peak = 200
            ydata = voltdata_array[peaks[i_peak]:peaks[i_peak]+width, i_ch]
            xdata = timedata_ltd[peaks[i_peak]:peaks[i_peak]+width]
            popt, pcov = curve_fit(func_exp_XOffset_wx0, xdata, ydata, bounds=([xdata[0], -np.inf, 0, 0], [xdata[1], np.inf, 1, 0.02]), p0=(xdata[0], -2.3, 0.2, 0.003))
            #axarr[0].plot(xdata, func_exp_XOffset_wx0(xdata, *popt), 'r-')
            #print(popt)
            '''
            try:
                ydata_2 = voltdata_array[peaks[i_peak]-width_2:peaks[i_peak]-pre_peak, i_ch]
                xdata_2 = timedata_ltd[peaks[i_peak]-width_2:peaks[i_peak]-pre_peak]
                popt_2, pcov_2 = curve_fit(func_LeakyReLU_like, xdata_2, ydata_2, bounds=([-np.inf, -np.inf, xdata_2[0], np.min(ydata_2)], [np.inf, np.inf, xdata_2[-1], np.max(ydata_2)]), p0=(0, 0.0001, xdata_2[-int(width_2/5)], ydata_2[0]))
                #axarr[0].plot(xdata_2, func_LeakyReLU_like(xdata_2, *popt_2), 'y-')
                #print(popt_2)
                #plt.plot(xdata_2, ydata_2)
                #plt.plot(xdata_2, func_LeakyReLU_like(xdata_2, *popt_2), 'g-')
                #plt.show()
                array_b.append(popt_2[1])
                array_x0.append(popt_2[2])
            except Exception as e:
                print(e)
            '''
            array_A.append(popt[2])
            array_tau.append(popt[3])
            perr_tau = np.sqrt(np.diag(pcov))
            array_err_tau.append(perr_tau[3])

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
                array_err_x0.append(perr[2])
                array_tau_ltd.append(array_tau[i])
                array_err_tau_ltd.append(array_err_tau[i])
            except Exception as e:
                print(e)
        idx_array_2 = np.where(np.array(array_b) < 0)
        array_b_ltd = np.delete(array_b, idx_array_2)
        array_x0_ltd = np.delete(array_x0, idx_array_2)
        array_err_x0_ltd = np.delete(array_err_x0, idx_array_2)
        array_tau_ltd = np.delete(array_tau_ltd, idx_array_2)
        array_err_tau_ltd = np.delete(array_err_tau_ltd, idx_array_2)

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


        '''
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
        axarr[3].plot(timedata_peaks_ltd, array_A_ltd, 'ob', label=label_A)
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
        filename = '%s_ch%d_raw_tau_A_%d' % (data_name_pxi, i_ch, self.ShotNo)
        #plt.savefig(filename)
        plt.show()
        '''

        '''
        plt.plot(xdata, ydata, 'b-')
        plt.plot(xdata, func_exp_XOffset(xdata, *popt), 'r-', label=label)
        plt.legend()
        plt.show()
        '''
        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

        return array_A_ltd, array_tau_ltd, array_b_ltd, array_x0_ltd, array_err_tau_ltd#array_err_x0_ltd#timedata_peaks_ltd

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
        for i_time in range(1340, 1600):
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
            mappable0 = axarr[0].pcolormesh(reff_a99_pci_sdens_rkt, k_pci_sdens_rkt, sdens_rkt[i_time].T, cmap='jet', norm=LogNorm(vmin=1e2, vmax=1e5))
            mappable1 = axarr[1].pcolormesh(reff_a99_pci_sdens_rvt, v_pci_sdens_rvt, sdens_rvt[i_time].T, cmap='jet', norm=LogNorm(vmin=1e-3, vmax=1e0))
            pp0 = fig.colorbar(mappable0, ax=axarr[0], orientation="vertical", pad=0.01)
            pp1 = fig.colorbar(mappable1, ax=axarr[1], orientation="vertical", pad=0.01)
            pp0.set_label('PSD', fontname="Arial", fontsize=10)
            pp1.set_label('PSD', fontname="Arial", fontsize=10)
            #plt.show()

            axarr[2].set_ylabel('reff/a99', fontsize=10)
            axarr[3].set_ylabel('reff/a99', fontsize=10)
            axarr[4].set_ylabel('Te [keV]', fontsize=10)
            axarr[4].set_xlabel('Time [sec]', fontsize=10)
            axarr[3].set_xlim(4.44, 4.70)
            axarr[2].set_xlim(4.44, 4.70)
            axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            axarr[3].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
            im0 = axarr[2].pcolormesh(t, reff_a99, ntilde_pci.T, cmap='jet',
                                      norm=LogNorm(vmin=1e5, vmax=5e6))
            im1 = axarr[3].pcolormesh(t, reff_a99, v_pci.T, cmap='jet')
            #im2 = axarr[2].pcolormesh(t, reff_a99, cosv_pci.T, cmap='jet', vmin=-0.5, vmax=0.5)
            pp0 = fig.colorbar(im0, ax=axarr[2], pad=0.01)
            pp1 = fig.colorbar(im1, ax=axarr[3], pad=0.01)
            pp0.set_label('ntilde[a.u.]', fontname="Arial", fontsize=10)
            pp1.set_label('v [km/sec]', fontname="Arial", fontsize=10)

            arr_erace = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 32, 33, 44,
                         45, 46, 47, 50, 51, 58]
            axarr[4].plot(t_ece, Te[:, arr_erace])
            axarr[4].set_xlim(4.44, 4.749)
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

        i_ch = 13#14#13#9
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

        _, _, _, timedata_peaks_ltd_,_ = self.findpeaks_radh(t_st, t_ed, i_ch)

        #timedata_peaks_ltd = timedata_peaks_ltd_
        #arr_peak_selection = [0,2,3,5,6,7]#169714
        #arr_peak_selection = [0,6,7,8,9,10,12,13]#169715
        #arr_peak_selection = [1,2,5,6,7,8]#169712
        #arr_peak_selection =[0,1,2,3,4,10,11] #169710
        #arr_peak_selection =[2,3,4] #169692
        #arr_peak_selection =[1,2,8] #169693
        #arr_peak_selection =[5,7,10] #169698
        #arr_peak_selection = [10] #169699
        #arr_peak_selection = [3,4,10] #169707
        #arr_peak_selection = [3,4,5,8,10,11,13] #169708
        timedata_peaks_ltd = timedata_peaks_ltd_[arr_peak_selection]
        #arr_toff = [-2e-4, 0, -2e-4, -3e-4, -3e-4, -2e-4, -1e-4]    #169710
        #arr_toff = [-2e-4, -3.5e-4, -3.5e-4, -1.5e-4, -2.5e-4, 0.5e-4]    #169712
        #arr_toff = 1e-4*np.array([-3,-1,0,-0.5,-1,-3,-2,-0.5])    #169715
        #arr_toff = 1e-4*np.array([-1.2,0,-1,-0.5,-1,-1])    #169714
        #arr_toff = 1e-4*np.array([-1,-3.5,-4])    #169692
        #arr_toff = 1e-4*np.array([-3,-2,-2])    #169693
        #arr_toff = 1e-4*np.array([-2,-1,0])    #169698
        #arr_toff = 1e-4*np.array([-0.5])    #169699
        #arr_toff = 1e-4*np.array([1.2,-2.2,-1.0])    #169707
        #arr_toff = 1e-4*np.array([-0.6,-0.9,0.1,-1,-2,-3.2,0])    #169708
        timedata_peaks_ltd -= arr_toff
        print(len(timedata_peaks_ltd_))
        t_st_condAV = 2e-3
        t_ed_condAV = 10e-3#6e-3
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
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Time [sec]')
        ax2.set_ylabel('b_dot_tilda [T/s]', rotation=270)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.show()
        '''


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
        #arr_peak_selection =[1,2,8] #169693
        #arr_toff = 1e-4*np.array([-3,-2,-2])    #169693
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
        #a = ax2.plot(buf_t_highK, 1e3*cond_av_highK, 'b-', label='highK', zorder=3)
        a = ax2.plot(buf_t_highK, 1e3*y2_highK, 'b-', label='highK', zorder=3)
        plt.title('Low-pass(<%dHz), #%d, %d' % (fs, ShotNo_1, ShotNo_2), loc='right')
        ax1.grid(axis='x', b=True, which='major', color='#666666', linestyle='-')
        ax1.grid(axis='x', b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax1.set_xlabel('Time [sec]')
        ax2.set_ylabel('Amplitude of e-scale turb.[a.u.]\nb_dot_tilda [T/s]', rotation=270, labelpad=15)
        ax1.set_ylabel('Intensity of ECE [a.u.]', labelpad=15)
        ax1.set_xlim(np.min(buf_t_radh), np.max(buf_t_radh))
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.show()

        '''
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
        '''

        self.findpeaks_radh(t_st=4.3, t_ed=4.6, i_ch=14, timedata=buf_t_radh+1, voltdata_array=cond_av_radh, isCondAV=True)

        filename = '_%d_%d.txt' % (ShotNo_1, ShotNo_2)
        filename_filtered = '_LP%dHz_%d_%d.txt' % (fs ,ShotNo_1, ShotNo_2)
        filename_radh_filtered = '_ch%d_LP%dHz_%d_%d.txt' % (i_ch+1, fs ,ShotNo_1, ShotNo_2)
        #np.savetxt('radh_array_condAV' + filename, cond_av_radh)
        #np.savetxt('radh_timedata_condAV' + filename, buf_t_radh)
        #np.savetxt('highK_array_condAV' + filename, cond_av_highK)
        #np.savetxt('highK_timedata_condAV' + filename, buf_t_highK)
        #np.savetxt('mp_array_condAV' + filename, cond_av_mp)
        #np.savetxt('mp_timedata_condAV' + filename, buf_t_mp)
        np.savetxt('radh_array_condAV' + filename_radh_filtered, y2)
        np.savetxt('highK_array_condAV' + filename_filtered, y2_highK)
        np.savetxt('mp_array_condAV' + filename_filtered, y2_mp)

        return buf_t_radh, cond_av_radh



    def ana_plot_radh_highK(self, t_st=None, t_ed=None):
        data_name_highK = 'mwrm_highK_Power3' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        data = AnaData.retrieve(data_name_highK, self.ShotNo, 1)
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        t = data.getDimData('Time')

        reff_a99_highK = data.getValData('reff/a99')
        #plt.plot(t, reff_a99_highK)
        #plt.show()
        data_highK = data.getValData('Amplitude (20-200kHz)')

        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)
        r = data_radhinfo.getValData('rho')

        i_ch = 9#14#13
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

        _, _, _, timedata_peaks_ltd,_ = self.findpeaks_radh(t_st, t_ed, i_ch)
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


        a = ax1.plot(t, 1e3*data_highK, 'b-', label='highK')
        b = ax1.plot(t_mp, data_mp_2d[:,0], 'g-', label='mp_raw')
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
        #plt.xlim(3.0, 5.3)
        plt.xlim(4.40, 4.48)
        #plt.xlim(4.398787, 4.403755)
        ax1.set_ylim(-5,30)
        #ax2.set_ylim(-3.4,-2.4)
        #plt.savefig("timetrace_radh_highK_mp_t439to440_No%d.png" % self.ShotNo)
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
        timedata = TimeData.retrieve(data_name_pxi, self.ShotNo, 1, 1).get_val()
        idx_time_st = find_closest(timedata, t_st)
        idx_time_ed = find_closest(timedata, t_ed)
        voltdata_array = np.zeros((len(timedata[idx_time_st:idx_time_ed]), 32))
        fs = 1e3
        dt = timedata[1] - timedata[0]
        freq = fftpack.fftfreq(len(timedata[idx_time_st:idx_time_ed]), dt)
        fig = plt.figure(figsize=(8,6), dpi=150)
        for i in range(32):
            voltdata = VoltData.retrieve(data_name_pxi, self.ShotNo, 1, i+1).get_val()
            '''
            low-pass filter for voltdata
            '''
            yf = fftpack.rfft(voltdata[idx_time_st:idx_time_ed]) / (int(len(voltdata[idx_time_st:idx_time_ed]) / 2))
            yf2 = np.copy(yf)
            yf2[(freq > fs)] = 0
            yf2[(freq < 0)] = 0
            y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata[idx_time_st:idx_time_ed]) / 2)))
            voltdata_array[:, i] = y2
            voltdata_array[:, i] = voltdata[idx_time_st:idx_time_ed]
            plt.plot(timedata[idx_time_st:idx_time_ed], voltdata[idx_time_st:idx_time_ed])
            label = 'R=%.3fm' % r[i]
            plt.plot(timedata[idx_time_st:idx_time_ed], y2, label=label)
        plt.xlim(4.4, 4.6)
        plt.legend()
        plt.xlabel('Time [sec]')
        title = '%s, #%d' % (data_name_pxi, self.ShotNo)
        plt.title(title, fontsize=18, loc='right')
        plt.show()

        idex_offset = find_closest(timedata, 4.475)

        fig = plt.figure(figsize=(8,6), dpi=150)
        title = 'Low-pass(<%dHz), #%d' % (fs, self.ShotNo)
        plt.title(title, fontsize=18, loc='right')
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr')
        #plt.pcolormesh(timedata, r, (voltdata_array - np.mean(voltdata_array[idex_offset:idex_offset+10000,:], axis=0)).T, cmap='bwr', vmin=-0.3, vmax=0.3)
        #plt.pcolormesh(timedata[:-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        plt.pcolormesh(timedata[idx_time_st:idx_time_ed-1], r, (voltdata_array[1:, :]-voltdata_array[:-1, :]).T, cmap='bwr', vmin=-0.0005, vmax=0.0005)
        #plt.pcolormesh(t, r, Te.T)
        #plt.pcolormesh(t, r[1:-1], (Te[:, 2:]-Te[:, 1:-1]-Te[:, 1:-1]+Te[:, :-2]).T, cmap='bwr')
        cb = plt.colorbar()
        #cb.set_label("d2Te/dr2")
        #cb.set_label(r"$V_{ECE}$ - offset")
        cb.set_label("dTe/dt")
        plt.xlabel("Time [sec]")
        #plt.xlim(4.4, 4.6)
        plt.ylabel("R [m]")
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        plt.show()
        filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        np.savetxt('radh_array' + filename, voltdata_array)
        np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

    def ana_plot_ece(self):
        data = AnaData.retrieve('ece_fast', self.ShotNo, 1)
        #data = AnaData.retrieve('ece_fast_calThom', self.ShotNo, 1)

        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
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
        rho = data.getValData('rho_vacuum')

        data = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        fig = plt.figure(figsize=(5, 2.5), dpi=250)
        plt.title(self.ShotNo, fontsize=12, loc='right')
        arr_erace = [5, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,32,33,44,45,46,47,50,51,58]
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
        plt.xlim(4.4, 4.65)
        plt.ylim(0.38, 0.7)
        #plt.ylim(3.85, 4.2)
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=10)
        plt.show()

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
        axarr[1].set_ylim(0, 7.5)
        #axarr[0].set_xlim(4.3, 4.8)
        #axarr[0].set_xlim(2.8, 6.0)
        #axarr[0].set_xlim(3.3, 5.3)
        #axarr[1].set_xlim(4.4, 4.65)
        axarr[0].set_xlim(4.4, 4.6)
        #axarr[0].set_xlim(3, 6)
        #axarr[1].set_xlim(3, 6)
        #axarr[0].set_ylim(0, )
        #axarr[1].set_ylim(0, )
        #axarr[0].set_xticklabels([])
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)

        arr_erace = [5, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,32,33,44,45,46,47,50,51,58]
        axarr[1].plot(t, Te[:, arr_erace])
        for i in range(vnum):
            ne_bar = data.getValData(i)
            axarr[0].plot(t_fir, ne_bar)
        f.align_labels()
        plt.show()
        #plt.savefig("timetrace_ECE_FIR_No%d.png" % self.ShotNo)

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

    def ana_plot_discharge_waveform4ITB(self):
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
        #title_shotinfo = (r'$B=%.3fT, R_{ax}=%.3fm, \gamma =%.4f, B_q=%.1f$') % (data_array_shotinfo[7], data_array_shotinfo[8], data_array_shotinfo[9], data_array_shotinfo[10])
        title_shotinfo_1 = r'$B={B:.3f}T, $'.format(B=data_array_shotinfo[7])
        title_shotinfo_2 = r'$={Rax_vac:.3f}m, \gamma={gamma:.4f}, B_q={Bq:.1f}%$'.format(Rax_vac=data_array_shotinfo[8], gamma=data_array_shotinfo[9], Bq=data_array_shotinfo[10])
        title_shotinfo = title_shotinfo_1 + r'$B_{ax}$' + title_shotinfo_2 + '%'
        title_date = '{year:d}-{month:02,d}-{day:02,d} {hour:d}:{minute:02,d}'.format(year=int(data_array_shotinfo[2]), month=int(data_array_shotinfo[3]), day=int(data_array_shotinfo[4]), hour=int(data_array_shotinfo[5]), minute=int(data_array_shotinfo[6]))
        title = ('#%d') % (self.ShotNo)
        vnum_ECH = data_ECH.getValNo()
        axarr[0].set_title(title, loc='left', fontsize=16)
        axarr[0].set_title(title_date + '\n' + title_shotinfo, loc='right', fontsize=12)
        axarr[2].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        view_ECH = np.array([12, 0, 1, 2, 4, 6])
        #view_ECH = np.array([12, 0, 1, 2, 4, 5, 6, 7])
        for i in range(len(view_ECH)):
            data_ = data_ECH.getValData(view_ECH[i])
            axarr[4].plot(t_ECH, data_, '-', label=data_ECH.getValName(view_ECH[i]), color=self.arr_color[i])

        arr_data_name = ['nb1', 'nb2', 'nb3', 'nb4a', 'nb4b', 'nb5a', 'nb5b']
        for i in range(len(arr_data_name)):
            data_NBI = AnaData.retrieve(arr_data_name[i] + 'pwr_temporal', self.ShotNo, 1)
            data_NBI_ = data_NBI.getValData(1)
            t_NBI = data_NBI.getDimData('time')
            label = str.upper(arr_data_name[i])
            axarr[3].plot(t_NBI, data_NBI_, label=label, color=self.arr_color[i])

        data_wp_ = data_wp.getValData('Wp')
        data_fir_ = data_fir.getValData(0)
        data_ip_ = data_ip.getValData('Ip')
        axarr[0].plot(t_Wp, data_wp_, color='black')
        axarr[1].plot(t_fir, data_fir_, color='red')
        axarr[2].plot(t_ip, data_ip_, label='Ip', color='blue')
        axarr[0].set_xlim(3, 6)
        axarr[0].set_ylim(0,)
        axarr[1].set_ylim(0,)
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
        plt.show()
        #plt.savefig("timetrace_%d.png" % self.ShotNo)

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


    def plot_TS(self):
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
        kernel = np.full((3,3), 1/9)
        #kernel = np.full((1,1), 1/1)
        #arr_all = movingaverage(data_2d=Te, axis=0, windowsize=5)
        arr_all = signal.convolve2d(Te, kernel, boundary='symm', mode='same')
        arr_all_dTe = signal.convolve2d(dTe, kernel, boundary='symm', mode='same')
        arr_R = r
        arr_1 = reff
        arr_dR = arr_R[1:] - arr_R[:-1]
        arr_dR_s = np.squeeze(arr_dR)
        arr_2 = (arr_all[:,1:] - arr_all[:, :-1]).T
        arr_dTedR = arr_2.T/arr_dR_s


        fig = plt.figure(figsize=(8,2), dpi=150)
        plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加

        '''
        X, _ = np.meshgrid(t, reff[0, :])
        #plt.pcolormesh(hs_thomson['Time'], hs_thomson['R'], arr_all.T, cmap='jet', vmin=0, vmax=6)
        #plt.pcolormesh(X, arr_1.T, arr_all.T, cmap='jet', vmin=0, vmax=10)
        #plt.pcolormesh((arr_all[:,1:] - arr_all[:, :-1]).T, cmap='bwr', vmin=-1, vmax=1)
        #plt.pcolormesh(X[1:, :], arr_1[:, 1:].T, arr_2, cmap='bwr', vmin=-1, vmax=1)
        #plt.pcolormesh(X[1:, :], arr_1[:, 1:].T, arr_dTedR.T, cmap='bwr', vmin=-100, vmax=100)
        plt.pcolormesh(X, arr_1.T, arr_all.T, cmap='jet', vmin=0, vmax=15)
        #plt.pcolormesh(np.array(tsmap_calib['Time']), np.array(tsmap_calib['reff/a99']), (arr_all[:,1:] - arr_all[:, :-1]).T, cmap='bwr', vmin=-1, vmax=1)
        #plt.vlines(4.398907, -1, 1, colors='red')
        #plt.vlines(4.399857, -1, 1, colors='blue')
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.xlabel('Time [sec]')
        #plt.ylim(-1, 1)
        #plt.ylabel('R [m]')
        plt.xlim(3.3, 5.3)
        plt.ylabel('$r_{eff}/a_{99}$', fontsize=15)
        #plt.ylim(2.9, 4.50)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.get_yaxis().labelpad = 15
        #cbar.set_label('$dT_e/dR$ [keV/m]', rotation=270)
        cbar.set_label('$T_e$ [keV]', rotation=270)
        idx_time = np.arange(86)
        idx_max = np.argmax(arr_dTedR[:, 30:70], axis=1)
        idx_min = np.argmin(arr_dTedR[:, 30:70], axis=1)
        reffa99_max = arr_1[idx_time, idx_max+31]
        reffa99_min = arr_1[idx_time, idx_min+31]
        plt.plot(t, reffa99_max, 'ro', markersize=2)
        plt.plot(t, reffa99_min, 'bo', markersize=2)
        plt.xlim(3.3, 5.3)
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
        #plt.xlim(-1.2, 1.2)
        #plt.xlim(0, 1.0)
        plt.xlabel('$r_{eff}/a_{99}$')
        plt.ylabel('$T_e [keV]$')
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.show()

        '''
        fig = plt.figure(figsize=(8,2), dpi=150)
        plt.plot(t, np.max(arr_dTedR[:, 30:90], axis=1), 'r-', label='reff/a99<0')
        plt.plot(t, np.abs(np.min(arr_dTedR[:, 30:90], axis=1)), 'b-', label='reff/a99>0')
        plt.xlim(3.3, 5.3)
        plt.ylim(0,150)
        plt.title('SN' + str(self.ShotNo), loc='right')
        plt.ylabel('Max($dT_e/dR$) \n[keV/m]')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=8)
        plt.show()
        '''

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

    def ana_plot_stft(self):
        #data = AnaData.retrieve('mp_raw_p', self.ShotNo, 1)
        data = AnaData.retrieve('mp_raw_t', self.ShotNo, 1)
        #data = AnaData.retrieve('mp_raw_t_1ch', self.ShotNo, 1)
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_2d = np.zeros((data.getDimSize(0), data.getDimSize(1)))

        data_2d = data.getValData(0)
        t = data.getDimData('TIME')
        #angle = data.getDimData('POL_ANGLE')
        angle = data.getDimData(0)
        #plt.plot(t, data_2d)
        #plt.xlim(5, 5.001)
        #plt.ylim(-0.7, 0.7)
        #plt.show()
        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        nperseg = int(input('Enter nperseg: '))
        noverlap = int(input('Enter noverlap : '))
        nfft = int(input('Enter nfft : '))
        cof_vmax = float(input('Enter cof_vmax: '))
        if nperseg == 0:
            nperseg = 2**10
        if noverlap == 0:
            noverlap = nperseg // 8
        if nfft == 0:
            nfft = nperseg
        if cof_vmax == 0:
            cof_vmax = 0.001
        start = time.time()

        y = data_2d[:, 0]
        x = t
        time_offset = x[0]

        MAXFREQ = 1e0
        N = 1e0 * np.abs(1 / (x[1] - x[2]))
        print('start spectgram')
        f, t, Zxx = sig.spectrogram(y, fs=N, nperseg=nperseg, noverlap=noverlap, nfft=nfft)#, window='hamming', nperseg=nperseg, mode='complex')
        print('end spectgram')
        #vmax = 0.05*np.max(Zxx)
        vmax = cof_vmax*np.max(Zxx)
        f /= 1e3


        print('start plot')
        ncols = 4
        nrows = 4
        grid = GridSpec(nrows, ncols, left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=0.3, hspace=0.3)
        fig = plt.figure(figsize=(8,6), dpi=150)
        ax1 = fig.add_subplot(grid[0:3, 0:3])
        plt.title('#%d' % self.ShotNo, loc='right')
        ax2 = fig.add_subplot(grid[3, 0:3])
        ax3 = fig.add_subplot(grid[0:3, 3])

        #fig = plt.figure()
        #ax1 = fig.add_subplot(311)
        #plt.title('#%d' % shotNo, loc='right')
        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(313)
        print('start pcolormesh')
        ax1.pcolormesh(t + time_offset, f, np.abs(Zxx), vmin=0, vmax=vmax, cmap='inferno')
        print('start xyplot')
        #ax2.plot(x[::10], y[::10])
        ax2.plot(x, y)
        print('start mean stft')
        ax1.set_ylabel("Frequency [kHz]")
        ax2.set_xlabel("Time [sec]")
        ax3.set_xscale('log')
        #ax1.set_xlim([np.min(x), np.max(x)])
        #ax2.set_xlim([np.min(x), np.max(x)])
        ax1.set_xlim([4.4, 4.6])
        ax2.set_xlim([4.4, 4.6])
        ax3.set_ylim([np.min(f), np.max(f)])
        print('start show')
        #fig.tight_layout()
        plt.show()
        #plt.savefig('plot.png')
        print('end plot')
        t_end = time.time() - start
        print("erapsed time for STFT = %.3f sec" % t_end)

        #t += time_offset
        #return x, y, f, t, Zxx

    def ana_plot_fluctuation(self):
        data = AnaData.retrieve('fmp_psd', self.ShotNo, 1)
        #data = AnaData.retrieve('pci_ch4_fft', self.ShotNo, 1)
        #data_pci_fast = AnaData.retrieve('pci_ch4_fft_fast', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        data_3d = np.zeros((data.getDimSize(0), data.getDimSize(1), vnum))

        for i in range(vnum):
            data_3d[:,:,i] = data.getValData(i)

        '''
        vnum = data_pci_fast.getValNo()
        t_pci_fast = data_pci_fast.getDimData('Time')
        data_0 = data_pci_fast.getValData(1)
        N = 1e0 * np.abs(1 / (t_pci_fast[1] - t_pci_fast[2]))
        print('start spectgram')
        f, t, Zxx = sig.spectrogram(data_0, fs=N)#, nperseg=nperseg, noverlap=noverlap, nfft=nfft)#, window='hamming', nperseg=nperseg, mode='complex')
        plt.pcolormesh(t, f, np.abs(Zxx))#, vmin=0, vmax=vmax)
        plt.show()
        '''


        #plt.imshow(data_3d[idx_time, :, :].T, cmap='jet')
        #plt.ylim(-0.5,3.5)
        #plt.xlim(0,99)
        #plt.imshow(data_3d[:, :, 1].T, cmap='jet')
        fig, axarr = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8,9), dpi=150)
        t = data.getDimData('Time')
        f = data.getDimData('Frequency')

        axarr[0].set_title("#%d" % self.ShotNo, fontsize=20, loc='left')
        axarr[0].set_ylabel('Frequency(PSD) [kHz]', fontsize=10)
        axarr[1].set_ylabel('Frequency(TORMODE) [kHz]', fontsize=10)
        axarr[2].set_ylabel('Frequency(COH) [kHz]', fontsize=10)
        axarr[2].set_xlabel('Time [sec]', fontsize=12)
        axarr[0].set_xlim(3, 6.5)
        axarr[1].set_xlim(3, 6.5)
        axarr[2].set_xlim(3, 6.5)
        axarr[0].set_ylim(0, 50)
        axarr[1].set_ylim(0, 50)
        axarr[2].set_ylim(0, 50)
        axarr[0].set_xticklabels([])
        axarr[1].set_xticklabels([])
        im0 = axarr[0].pcolormesh(t, f, data_3d[:, :, 0].T, cmap='jet', vmax=1e-9)
        im1 = axarr[1].pcolormesh(t, f, data_3d[:, :, 1].T, cmap='jet')
        im2 = axarr[2].pcolormesh(t, f, data_3d[:, :, 3].T, cmap='jet')
        fig.colorbar(im0, ax=axarr[0], pad=0.01)
        fig.colorbar(im1, ax=axarr[1], pad=0.01)
        fig.colorbar(im2, ax=axarr[2], pad=0.01)
        #plt.savefig("fmp_psd_No%d.png" % self.ShotNo)
        plt.show()

def ana_delaytime_shotarray():
    shot_array = 167080 + np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17])
    #shot_array = 163950 + np.array([0, 1, 2, 3, 4, 5, 6, 8])
    try:
        for i, shotno in enumerate(shot_array):
            itba = ITB_Analysis(shotno)
            itba.ana_delaytime_radh_allch(t_st=4.4, t_ed=4.6)
            print(shotno)

    except Exception as e:
        print(e)

def ana_findpeaks_shotarray():
    #shot_array = 167080 + np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17])
    shot_array = 163950 + np.array([0, 1, 2, 3, 4, 5, 6, 8])
    t_st = 4.4
    t_ed = 4.6
    data_name_pxi = 'RADHPXI'
    for i_ch in range(32):
        array_A = []
        array_tau = []
        array_b = []
        try:
            for i, shotno in enumerate(shot_array):
                itba = ITB_Analysis(shotno)
                buf_array_A, buf_array_tau, buf_array_b, _,_ = itba.findpeaks_radh(t_st=t_st, t_ed=t_ed, i_ch=i_ch)
                array_A.extend(buf_array_A)
                array_tau.extend(buf_array_tau)
                array_b.extend(buf_array_b)


            mean_A = statistics.mean(array_A)
            stdev_A = statistics.stdev(array_A)
            mean_tau = statistics.mean(array_tau)
            stdev_tau = statistics.stdev(array_tau)
            mean_b = statistics.mean(array_b)
            stdev_b = statistics.stdev(array_b)

            label_A = 'mean:%.6f, stdev:%.6f' % (mean_A, stdev_A)
            label_tau = 'mean:%.6f, stdev:%.6f' % (mean_tau, stdev_tau)
            label_b = 'mean:%.6f, stdev:%.6f' % (mean_b, stdev_b)

            f, axarr = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), dpi=150, sharex=True)
            title = '%s(ch.%d), #%d-%d' % (data_name_pxi, i_ch, shot_array[0], shot_array[-1])
            axarr[0].set_title(title, fontsize=18, loc='right')
            axarr[0].plot(array_A, 'bo', label=label_A)
            axarr[1].plot(array_tau, 'ro', label=label_tau)
            axarr[2].plot(array_b, 'go', label=label_b)
            axarr[2].set_xlabel('Event Number', fontsize=15)
            axarr[0].set_ylabel('A(0,1)', fontsize=15)
            axarr[1].set_ylabel('tau(0,1)', fontsize=15)
            axarr[2].set_ylabel('b(0,)', fontsize=15)
            axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[2].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
            axarr[0].legend(frameon=True, loc='upper left', fontsize=15)
            axarr[1].legend(frameon=True, loc='upper left', fontsize=15)
            axarr[2].legend(frameon=True, loc='upper left', fontsize=15)
            filename = '%s_ch%d_mean_stdev_A_tau_b_%dto%d' % (data_name_pxi, i_ch, shot_array[0], shot_array[-1])
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
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def low_pass_filter(timedata, voltdata, freq, fs):
    '''
    low-pass filter for voltdata
    '''
    yf = fftpack.rfft(voltdata) / (int(len(voltdata) / 2))
    yf2 = np.copy(yf)
    yf2[(freq > fs)] = 0
    yf2[(freq < 0)] = 0
    y2 = np.real(fftpack.irfft(yf2) * (int(len(voltdata) / 2)))

def func_exp_XOffset(x, y0, A, tau):
    return y0 + A*np.exp(-x/tau)

def func_exp_XOffset_wx0(x, x0, y0, A, tau):
    return y0 + A*np.exp(-(x-x0)/tau)

def func_LeakyReLU_like(x, a, b, x0, y0):
    return np.where(x < x0, a*(x-x0) + y0, b*(x-x0) + y0)

def test_function():
    x = np.linspace(-1, 1, 100)
    y = func_LeakyReLU_like(x, 0.01, 0.1, -0.1, -0.2)
    plt.plot(x, y)
    plt.show()

def movingaverage(data_2d,axis=1,windowsize=3):
    answer = np.zeros((data_2d.shape))
    answer[:,:] = np.nan

    v = np.ones(windowsize,)/windowsize

    for i in range(data_2d.shape[axis]):
        if axis==0:
            answer[i,windowsize-1:]=np.convolve(data_2d[i,:], v, mode = "valid")
        if axis==1:
            answer[windowsize-1:,i]=np.convolve(data_2d[:,i], v, mode = "valid")
        answer=answer
    return answer


if __name__ == "__main__":
    start = time.time()
    #for i in range(163943, 163975):
    #    itba = ITB_Analysis(i)
    #    #itba.ana_plot_ece()
    #    itba.ana_plot_discharge_waveform4ITB()
    #    itba.ana_plot_fluctuation()
    #    itba.ana_plot_allTS_sharex()
    #ShotNo = input('Enter Shot Number: ')
    #ShotNo = sys.argv[1]
    #ana_findpeaks_shotarray()
    #ana_delaytime_shotarray()
    #itba = ITB_Analysis(int(ShotNo))
    itba = ITB_Analysis(169710, 169712)#167088), 163958
    #itba.ana_plot_highSP_TS(0)
    #itba.get_ne(target_t=4.4, target_r=0.2131075)
    #itba.ana_plot_ece()
    #itba.ana_plot_radh(t_st=4.40, t_ed=4.50)
    #itba.plot_TS()
    #itba.plot_2TS()
    #itba.ana_plot_pci_test()
    #itba.conditional_average_highK(t_st=4.3, t_ed=4.6)
    itba.combine_shots_data()
    #itba.findpeaks_radh(t_st=4.4, t_ed=4.6, i_ch=26)
    #itba.find_peaks_radh_allch(t_st=4.4, t_ed=4.6)
    #itba.ana_delaytime_radh_allch(t_st=4.3, t_ed=4.6, isCondAV=True)
    #itba.ana_plot_stft()
    #itba.ana_plot_fluctuation()
    #itba.ana_plot_allMSE_sharex()
    #itba.ana_plot_discharge_waveform_basic()
    #itba.ana_plot_discharge_waveform4ITB()
    #itba.ana_plot_allTS_sharex()
    #itba.ana_plot_allThomson_sharex()
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[127704, 127705, 127709, 131617, 131618, 143883, 143895])
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[163958, 163959, 163960, 163963, 163964, 163965, 163966, 163967])
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[163958, 167090, 167091, 167092, 167093, 167094, 167095, 167096])
    #itba.ana_plot_discharge_waveforms(arr_ShotNo=[163958, 167097, 167098, 167099])
    t = time.time() - start
    print("erapsed time= %.3f sec" % t)
