import sys
from nifs.anadata import *
from ftplib import FTP
import scipy.signal as sig
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ITB_Analysis:
    def __init__(self, ShotNo):
        self.ShotNo = ShotNo

    def ana_plot_allTS_sharex(self):

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
        idx_arr_time_lhdgauss = find_closest(t_lhdgauss, t)
        val_arr_time = t[idx_arr_time]

        plt.clf()
        f, axarr = plt.subplots(3, len(t), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(2*len(t),6), dpi=100)

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
            axarr[1,idx].plot(r, T_fit,  '-', color='red')
            axarr[1,idx].errorbar(r, T, yerr=dT, capsize=2, fmt='o', markersize=2, ecolor='red', markeredgecolor='red', color='w')
            axarr[0,idx].errorbar(r, n, yerr=dn, capsize=2, fmt='o', markersize=2, ecolor='blue', markeredgecolor='blue', color='w')
            axarr[0,idx].plot(r, n_fit,  '-', color='blue')
            axarr[1,idx].set_xlim(-1.2, 1.2)
            axarr[0,idx].set_xlim(-1.2, 1.2)
            axarr[0,idx].set_ylim(0, 1.5)
            axarr[1,idx].set_ylim(0, 10.5)
            title_time = ('t=%.3fsec') % (t[idx])
            axarr[0,idx].set_title(title_time)

        try:
            for i, idx in enumerate(idx_arr_time_lhdgauss):
                sum_pd = sum_power_density[idx, :]
                sum_tp = sum_total_power[idx, :]
                axarr[2,i].plot(-reff_lhdgauss, sum_pd/2, '-', label='Power_Density', color='gray')
                axarr[2,i].plot(-reff_lhdgauss, sum_tp/2, '--', label='Total_Power', color='black')
                axarr[2,i].set_xlim(-1.2, 1.2)
                axarr[2,i].set_ylim(0, 1.2)
                axarr[2,i].set_xlabel(r'$r_{eff}/a_{99}$')

            for i, idx in enumerate(idx_arr_time):
                r_1d = reff_mse[i, :]
                iota_1d = iota[i, :]  # R 次元でのスライスを取得
                iota_vac_1d = iota_vac[i, :]  # R 次元でのスライスを取得
                iota_data_ip_1d = iota_data_ip[i, :]  # R 次元でのスライスを取得
                iota_data_error_ip_1d = iota_data_error_ip[i, :]  # R 次元でのスライスを取得
                axarr[2,idx].plot(r_1d, iota_1d, '-', label='iota', color='green')
                axarr[2,idx].errorbar(r_1d, iota_data_ip_1d, yerr=iota_data_error_ip_1d, capsize=2, fmt='o', markersize=2, ecolor='green', markeredgecolor='green', color='w')
                axarr[2,idx].plot(r_1d, iota_vac_1d, '--', label='iota_vac', color='gray')
        except:
            print('error')

        axarr[0, 0].set_ylabel(r'$n_e [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e [keV]$')
        axarr[2, 0].set_ylabel(r'$\iota/2\pi$')


        #plt.savefig("Te_ne_iota_No%d.png" % self.ShotNo)
        plt.show()
        plt.close()

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
        f, axarr = plt.subplots(3, len(t_mse), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(2*len(t_mse),6), dpi=150)

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
            axarr[1,i].set_xlim(-1.2, 1.2)
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
            axarr[2,i].set_xlim(-1.2, 1.2)
            axarr[2,i].set_ylim(0, 1.2)
            axarr[2,i].set_xlabel(r'$r_{eff}/a_{99}$')

        axarr[0, 0].set_ylabel(r'$n_e [10^{19} m^{-3}]$')
        axarr[1, 0].set_ylabel(r'$T_e [keV]$')
        axarr[2, 0].set_ylabel(r'$\iota/2\pi$')


        plt.show()
        plt.close()

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

    def ana_plot_ece(self):
        data = AnaData.retrieve('ece_fast', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        Te = data.getValData('Te')

        data = AnaData.retrieve('fir_nel', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t_fir = data.getDimData('Time')

        plt.clf()
        plt.close()
        f, axarr = plt.subplots(2, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(16, 18), dpi=150)

        axarr[0].set_title(self.ShotNo, fontsize=18, loc='right')
        axarr[0].set_ylabel(r'$T_e$', fontsize=18)
        axarr[1].set_xlabel(r'$Time [sec]$', fontsize=18)
        axarr[1].set_ylabel(r'$n_e$', fontsize=18)
        axarr[0].set_xlim(3, 6.5)
        axarr[1].set_xlim(3, 6.5)
        axarr[0].set_xticklabels([])
        for i in range(vnum):
            axarr[0].plot(t, Te)
            ne_bar = data.getValData(i)
            axarr[1].plot(t_fir, ne_bar)
        plt.show()
        #plt.savefig("timetrace_ECE_FIR_No%d.png" % self.ShotNo)

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
        fig, axarr = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(16, 18), dpi=150)
        t = data.getDimData('Time')
        f = data.getDimData('Frequency')

        axarr[0].set_title("#%d" % self.ShotNo, fontsize=18, loc='left')
        axarr[0].set_ylabel('Frequency(PSD) [Hz]', fontsize=8)
        axarr[1].set_ylabel('Frequency(TORMODE) [Hz]', fontsize=8)
        axarr[2].set_ylabel('Frequency(COH) [Hz]', fontsize=8)
        axarr[2].set_xlabel('Time [sec]', fontsize=8)
        axarr[0].set_xlim(3, 6.5)
        axarr[1].set_xlim(3, 6.5)
        axarr[2].set_xlim(3, 6.5)
        axarr[0].set_xticklabels([])
        axarr[1].set_xticklabels([])
        im0 = axarr[0].pcolormesh(t, f, data_3d[:, :, 0].T, cmap='jet', vmax=1e-10)
        im1 = axarr[1].pcolormesh(t, f, data_3d[:, :, 1].T, cmap='jet')
        im2 = axarr[2].pcolormesh(t, f, data_3d[:, :, 3].T, cmap='jet')
        fig.colorbar(im0, ax=axarr[0], pad=0.01)
        fig.colorbar(im1, ax=axarr[1], pad=0.01)
        fig.colorbar(im2, ax=axarr[2], pad=0.01)
        plt.show()

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

if __name__ == "__main__":
    itba = ITB_Analysis(143895)
    #itba.ana_plot_ece()
    #itba.ana_plot_allTS_sharex()
    itba.ana_plot_fluctuation()
