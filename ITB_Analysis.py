import sys
from nifs.anadata import *
from nifs.voltdata import *
from nifs.timedata import *
from matplotlib.gridspec import GridSpec
from ftplib import FTP
from scipy import fftpack
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.signal as sig
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import statistics


class ITB_Analysis:
    def __init__(self, ShotNo):
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

    def ana_delaytime_radh_allch(self, t_st, t_ed):
        data_name_info = 'radhinfo' #radlinfo
        data_name_pxi = 'RADHPXI' #RADLPXI
        data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data_radhinfo.getValData('rho')
        fig = plt.figure(figsize=(8,5), dpi=150)
        for i_ch in range(32):
            try:
                print(self.ShotNo, i_ch)
                _,_,timedata_peaks_ltd = self.findpeaks_radh(t_st, t_ed, i_ch)
                ydata = r[i_ch]*np.ones(len(timedata_peaks_ltd))
                plt.plot(timedata_peaks_ltd, ydata, 's')
            except Exception as e:
                print(e)

        title = '%s, #%d' % (data_name_pxi, self.ShotNo)
        plt.title(title)
        plt.xlabel('Time [sec]')
        plt.ylabel('r/a')
        plt.ylim(0,1)
        #plt.ylabel('R [m]')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        #filename = 'peaktime_%s_raw_tau_A_%d' % (data_name_pxi, self.ShotNo)
        filename = 'risetime_%s_raw_tau_A_%d_v2' % (data_name_pxi, self.ShotNo)
        plt.savefig(filename)
        #plt.show()


    def find_peaks_radh_allch(self, t_st, t_ed):
        for i_ch in range(32):
            try:
                self.findpeaks_radh(t_st, t_ed, i_ch)
            except Exception as e:
                print(e)


    def findpeaks_radh(self, t_st, t_ed, i_ch):
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

        f, axarr = plt.subplots(4, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8,8), dpi=150, sharex=True)

        axarr[0].set_xlim(t_st, t_ed)
        axarr[0].plot(timedata_ltd, voltdata_array[:, i_ch])
        axarr[0].plot(timedata_ltd[peaks], voltdata_array[peaks, i_ch], "or")
        title = '%s(ch.%d), #%d' % (data_name_pxi, i_ch, self.ShotNo)
        axarr[0].set_title(title, fontsize=18, loc='right')

        array_A = []
        array_tau = []
        array_b = []
        array_x0 = []

        for i_peak in range(len(peaks)):
            width = 5000
            #width_2 = 2000
            #pre_peak = 200
            ydata = voltdata_array[peaks[i_peak]:peaks[i_peak]+width, i_ch]
            xdata = timedata_ltd[peaks[i_peak]:peaks[i_peak]+width]
            popt, pcov = curve_fit(func_exp_XOffset_wx0, xdata, ydata, bounds=([xdata[0], -np.inf, 0, 0], [xdata[1], np.inf, 1, 0.02]), p0=(xdata[0], -2.3, 0.2, 0.003))
            axarr[0].plot(xdata, func_exp_XOffset_wx0(xdata, *popt), 'r-')
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

        timedata_peaks = timedata_ltd[peaks]
        idx_array = np.where((np.array(array_A) < 0.001) | (np.array(array_tau) > 0.019) | (np.array(array_tau) < 0.001))
        array_A_ltd = np.delete(array_A, idx_array)
        array_tau_ltd = np.delete(array_tau, idx_array)
        peaks_ltd = np.delete(peaks, idx_array)

        timedata_peaks_ltd = np.delete(timedata_peaks, idx_array)
        for _, i_peak_ltd in enumerate(peaks_ltd):
            width_2 = 2000
            pre_peak = 200
            try:
                ydata_2 = voltdata_array[i_peak_ltd-width_2:i_peak_ltd-pre_peak, i_ch]
                xdata_2 = timedata_ltd[i_peak_ltd-width_2:i_peak_ltd-pre_peak]
                popt_2, pcov_2 = curve_fit(func_LeakyReLU_like, xdata_2, ydata_2, bounds=([-np.inf, -np.inf, xdata_2[0], np.min(ydata_2)], [np.inf, np.inf, xdata_2[-1], np.max(ydata_2)]), p0=(0, 0.0001, xdata_2[-int(width_2/5)], ydata_2[0]))
                axarr[0].plot(xdata_2, func_LeakyReLU_like(xdata_2, *popt_2), 'y-')
                #print(popt_2)
                array_b.append(popt_2[1])
                array_x0.append(popt_2[2])
            except Exception as e:
                print(e)
            array_A.append(popt[2])
            array_tau.append(popt[3])
        idx_array_2 = np.where(np.array(array_b) < 0)
        array_b_ltd = np.delete(array_b, idx_array_2)
        array_x0_ltd = np.delete(array_x0, idx_array_2)


        mean_A = statistics.mean(array_A_ltd)
        stdev_A = statistics.stdev(array_A_ltd)
        mean_tau = statistics.mean(array_tau_ltd)
        stdev_tau = statistics.stdev(array_tau_ltd)
        mean_b = statistics.mean(array_b_ltd)
        stdev_b = statistics.stdev(array_b_ltd)

        label_A = 'mean:%.6f, stdev:%.6f' % (mean_A, stdev_A)
        label_tau = 'mean:%.6f, stdev:%.6f' % (mean_tau, stdev_tau)
        label_b = 'mean:%.6f, stdev:%.6f' % (mean_b, stdev_b)

        axarr[1].plot(array_x0_ltd, array_b_ltd, 'og', label=label_b)
        axarr[1].legend(frameon=True, loc='upper left', fontsize=15)
        axarr[1].set_ylabel('b', fontsize=15)
        #axarr[1].set_ylim(0, 2e-2)
        axarr[1].set_xlim(t_st, t_ed)
        axarr[2].plot(timedata_peaks_ltd, array_tau_ltd, 'or', label=label_tau)
        axarr[2].legend(frameon=True, loc='upper left', fontsize=15)
        #plt.title('tau(0,0.02)')
        axarr[2].set_ylabel('tau(0,0.02)', fontsize=15)
        axarr[2].set_ylim(0, 2e-2)
        axarr[2].set_xlim(t_st, t_ed)
        axarr[3].plot(timedata_peaks_ltd, array_A_ltd, 'ob', label=label_A)
        axarr[3].legend(frameon=True, loc='lower left', fontsize=15)
        #plt.title('A(0,1)')
        axarr[3].set_xlim(t_st, t_ed)
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
        plt.plot(xdata, ydata, 'b-')
        plt.plot(xdata, func_exp_XOffset(xdata, *popt), 'r-', label=label)
        plt.legend()
        plt.show()
        '''
        #filename = '_t%.2fto%.2f_%d.txt' % (t_st, t_ed, self.ShotNo)
        #np.savetxt('radh_array' + filename, voltdata_array)
        #np.savetxt('radh_timedata' + filename, timedata[idx_time_st:idx_time_ed])

        return array_A_ltd, array_tau_ltd, array_b_ltd, array_x0_ltd#timedata_peaks_ltd

    def ana_plot_mwrm_highK_Power(self, t_st, t_ed):
        data_name_pxi = 'm' #RADLPXI data_radhinfo = AnaData.retrieve(data_name_info, self.ShotNo, 1)

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
        #plt.xlim(4.4, 4.6)
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
        #plt.xlim(4.4, 4.6)
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

        data = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        '''
        plt.title(self.ShotNo, fontsize=18, loc='right')
        plt.pcolormesh(t[:-1], r, (Te[1:, :]-Te[:-1, :]).T, cmap='bwr')
        #plt.pcolormesh(t, r, Te.T)
        #plt.pcolormesh(t, r[1:-1], (Te[:, 2:]-Te[:, 1:-1]-Te[:, 1:-1]+Te[:, :-2]).T, cmap='bwr')
        cb = plt.colorbar()
        #cb.set_label("d2Te/dr2")
        cb.set_label("dTe/dt")
        plt.xlabel("Time [sec]")
        plt.ylabel("R [m]")
        plt.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        plt.show()
        '''
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t_fir = data.getDimData('Time')

        f, axarr = plt.subplots(2, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 10), dpi=150, sharex=True)

        axarr[0].set_title(self.ShotNo, fontsize=18, loc='right')
        axarr[0].set_ylabel(r'$T_e (ECE)$', fontsize=18)
        axarr[1].set_xlabel('Time [sec]', fontsize=18)
        axarr[1].set_ylabel(r'$n_eL\ [10^{19}m^{-2}]$', fontsize=18)
        axarr[0].set_ylim(0, 10.3)
        axarr[0].set_xlim(4.3, 4.8)
        #axarr[1].set_xlim(4.4, 4.7)
        #axarr[0].set_xlim(3, 6)
        #axarr[1].set_xlim(3, 6)
        #axarr[0].set_ylim(0, )
        #axarr[1].set_ylim(0, )
        #axarr[0].set_xticklabels([])
        axarr[0].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        axarr[1].tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True, labelsize=12)
        for i in range(vnum):
            axarr[0].plot(t, Te)
            ne_bar = data.getValData(i)
            axarr[1].plot(t_fir, ne_bar)
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
        view_ECH = np.array([12, 0, 1, 2, 4, 5, 6, 7])
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
    #shot_array = 167080 + np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17])
    shot_array = 163950 + np.array([0, 1, 2, 3, 4, 5, 6, 8])
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
                buf_array_A, buf_array_tau, buf_array_b, _ = itba.findpeaks_radh(t_st=t_st, t_ed=t_ed, i_ch=i_ch)
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
    itba = ITB_Analysis(169715)#167088), 163958
    #itba.ana_plot_highSP_TS(0)
    #itba.ana_plot_ece()
    itba.ana_plot_radh(t_st=4.3, t_ed=4.6)
    #itba.findpeaks_radh(t_st=4.4, t_ed=4.6, i_ch=26)
    #itba.find_peaks_radh_allch(t_st=4.4, t_ed=4.6)
    #itba.ana_delaytime_radh_allch(t_st=4.4, t_ed=4.6)
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
