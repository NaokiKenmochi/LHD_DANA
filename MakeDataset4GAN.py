#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tqdm
from nifs.anadata import *
from ftplib import FTP
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
#import scipy.signal as sig

class Normalization_factor:
    def __init__(self):
        self.vname = ''
        self.xmin = 0.0
        self.xmax = 0.0

class MakeDataset4GAN(Normalization_factor):
    def __init__(self, ShotNo, mode):
        self.ShotNo = ShotNo
        self.mode = mode
        if(self.mode != "deposition_calculation" and self.mode != "ECHsetting_estimation"):
            print("ERROR: mode must be deposition_calculation or ECHsetting_estimation")
            sys.exit()
        self.norm_fact= [["Te", 0, 10],
                        ["ne", 0,5],
                        ["RAX", 3.5,4.0],
                        ["Bq", -50, 200],
                        ["gamma", 1.129,1.262],
                        ["p0", 0, 10],
                        ["pf", 1.41, 5.00],
                        ["ip", -150, 150],
                        ["ipf", -12.60, 4.00],
                        ["power", 0, 2.0],
                        ["Rfocus", 3.3, 3.9],
                        ["Tfocus", -0.8, 1.2],
                        ["Zfocus", -0.5, 0.5],
                        ["alpha", -90, 90],
                        ["beta", -45, 45],
                        ["O_Power_Density", 0, 2],
                         ["O_Total_Power", 0, 2],
                         ["X_Power_Density", 0, 2],
                         ["X_Total_Power", 0, 2]]

        self.R_fir = [0, 0, 3.309, 3.399, 3.489, 3.579, 3.669, 3.759, 3.849, 3.939, 4.029, 4.119, 4.209, 4.299, 4.389]

    def make_dataset_line7(self, ShotNo):
        data_shotinfo = AnaData.retrieve('shotinfo', ShotNo, 1)
        RAX = float(data_shotinfo.getValData('Rax_vac'))
        Bq = float(data_shotinfo.getValData('Bq'))
        gamma = float(data_shotinfo.getValData('gamma'))

        data_tsmap_nel = AnaData.retrieve('tsmap_nel', ShotNo, 1)
        p0 = data_tsmap_nel.getValData('p0')
        pf = data_tsmap_nel.getValData('pf')
        ip = data_tsmap_nel.getValData('ip')
        ipf = data_tsmap_nel.getValData('ipf')
        data_LGdep = AnaData.retrieve('LHDGAUSS_DEPROF', ShotNo, 1)
        data_LGin = AnaData.retrieve('LHDGAUSS_INPUT', ShotNo, 1)
        data_TS = AnaData.retrieve('tsmap_smooth', ShotNo, 1)
        t_tsmap_nel = data_tsmap_nel.getDimData('Time')
        t_LGin = data_LGin.getDimData('time')
        t_LGdep = data_LGdep.getDimData('time')
        t_TS = data_TS.getDimData('Time')
        #reff = data.getValData('reff/a99')
        Te_fit = data_TS.getValData('Te_fit')
        ne_fit = data_TS.getValData('ne_fit')

        dsize_time = data_LGin.getDimSize(0)
        dsize_line = data_LGin.getDimSize(1)
        vnum_LGin = data_LGin.getValNo()
        data_3d_LGin = np.zeros((dsize_time, dsize_line, vnum_LGin))

        #normalization
        Te_fit = normalize(Te_fit, xmin=self.norm_fact[0][1], xmax=self.norm_fact[0][2])
        ne_fit = normalize(ne_fit, xmin=self.norm_fact[1][1], xmax=self.norm_fact[1][2])
        RAX = normalize(RAX , xmin=self.norm_fact[2][1], xmax=self.norm_fact[2][2])
        Bq = normalize(Bq, xmin=self.norm_fact[3][1], xmax=self.norm_fact[3][2])
        gamma = normalize(gamma, xmin=self.norm_fact[4][1], xmax=self.norm_fact[4][2])
        p0 = normalize(p0, xmin=self.norm_fact[5][1], xmax=self.norm_fact[5][2])
        pf = normalize(pf, xmin=self.norm_fact[6][1], xmax=self.norm_fact[6][2])
        ip = normalize(ip, xmin=self.norm_fact[7][1], xmax=self.norm_fact[7][2])
        ipf = normalize(ipf, xmin=self.norm_fact[8][1], xmax=self.norm_fact[8][2])
        for i in range(vnum_LGin):
            data_3d_LGin[:,:,i] = data_LGin.getValData(i)
        vnum_LGdep = data_LGdep.getValNo()
        data_3d_LGdep = np.zeros((data_LGdep.getDimSize(0), data_LGdep.getDimSize(1), vnum_LGdep))
        for i in range(vnum_LGdep):
            data_3d_LGdep[:,:,i] = data_LGdep.getValData(i)

        dpi = 100
        figsize_px = np.array([100, 100])
        figsize_inch = figsize_px / dpi
        for i_time in range(dsize_time):
            #for i_line in range(dsize_line):
            i_line = 6
            i_tank = 6
            if(data_3d_LGin[i_time, i_line, 3]>0.0 and data_3d_LGin[i_time, i_line, 2]==154):    #get data if Pech_inj > 0.0 MW and freq==154
                arr_input_2D = np.zeros((100, 100))
                idx_time_TS = getNearestIndex(t_TS, t_LGin[i_time])
                idx_time_LGdep = getNearestIndex(t_LGdep, t_LGin[i_time])
                idx_time_tsmap_nel = getNearestIndex(t_tsmap_nel, t_LGin[i_time])
                t_LGin_round3 = round(t_LGin[i_time], 3)
                t_LGdep_round3 = round(t_LGdep[idx_time_LGdep], 3)
                t_TS_round3 = round(t_TS[idx_time_TS], 3)
                #if(t_LGin[i_time]==t_LGdep[idx_time_LGdep] and t_LGin[i_time]==t_TS[idx_time_TS]):
                if(t_LGin_round3==t_LGdep_round3 and t_LGin_round3==t_TS_round3):
                    Te = Te_fit[idx_time_TS, :]
                    ne = ne_fit[idx_time_TS, :]
                    if(self.mode == "deposition_calculation"):
                        arr_input_2D[:8, :81] = Te[:81]
                        arr_input_2D[8:16, :80] = Te[81:]
                        arr_input_2D[16:24, :81] = ne[:81]
                        arr_input_2D[24:32, :80] = ne[81:]
                        arr_input_2D[35:40, :] = RAX    #RAX [m]
                        arr_input_2D[40:45, :] = Bq     #Bq[%]
                        arr_input_2D[45:50, :] = gamma  #gamma['']
                        arr_input_2D[50:55, :] = p0[idx_time_tsmap_nel]    #p0 [%]
                        arr_input_2D[55:60, :] = pf[idx_time_tsmap_nel]    #pf [arb]
                        arr_input_2D[60:65, :] = ip[idx_time_tsmap_nel]    #ip [kA/T]
                        arr_input_2D[65:70, :] = ipf[idx_time_tsmap_nel]    #ipf [arb]
                        arr_input_2D[70:75, :] = normalize(data_3d_LGin[i_time, i_line, 3], xmin=self.norm_fact[9][1], xmax=self.norm_fact[9][2])    #power [MW]
                        arr_input_2D[75:80, :] = normalize(data_3d_LGin[i_time, i_line, 4], xmin=self.norm_fact[10][1], xmax=self.norm_fact[10][2])    #Rfocus [m]
                        arr_input_2D[80:85, :] = normalize(data_3d_LGin[i_time, i_line, 5], xmin=self.norm_fact[11][1], xmax=self.norm_fact[11][2])    #Tfocus [m]
                        arr_input_2D[85:90, :] = normalize(data_3d_LGin[i_time, i_line, 6], xmin=self.norm_fact[12][1], xmax=self.norm_fact[12][2])    #Zfocus [m]
                        arr_input_2D[90:95, :] = normalize(data_3d_LGin[i_time, i_line, 13], xmin=self.norm_fact[13][1], xmax=self.norm_fact[13][2])   #alpha [deg]
                        arr_input_2D[95:, :] = normalize(data_3d_LGin[i_time, i_line, 14], xmin=self.norm_fact[14][1], xmax=self.norm_fact[14][2])     #beta [deg]

                        fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
                        ax1.axis("off")
                        ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False,
                                        labelleft=False,
                                        labelright=False, labeltop=False)
                        fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)
                        ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                        filepath = "ECHdeposition_est_Pmax2MW_line7/input"
                        filename = "dataset4GAN_LHDGAUSS_SN%d_t%.3fs_ECHline%d.png" % (ShotNo, t_LGin[i_time], i_line+1)
                        #fig1.savefig(filepath +"/"+ filename, bbox_inches="tight", pad_inches=-0.033)
                        plt.close(fig1)

                        arr_input_2D[:25,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 2+4*i_tank], xmin=self.norm_fact[15][1], xmax=self.norm_fact[15][2])
                        arr_input_2D[25:50,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 3+4*i_tank], xmin=self.norm_fact[16][1], xmax=self.norm_fact[16][2])
                        arr_input_2D[50:75,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 4+4*i_tank], xmin=self.norm_fact[17][1], xmax=self.norm_fact[17][2])
                        arr_input_2D[75:,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 5+4*i_tank], xmin=self.norm_fact[18][1], xmax=self.norm_fact[18][2])
                        ax1.imshow(arr_input_2D, vmin=0, vmax=1,interpolation='none', cmap='PuRd')
                        filepath = "ECHdeposition_est_Pmax2MW_line7/output"
                        filename = "dataset4GAN_LHDGAUSS_SN%d_t%.3fs_ECHline%d.png" % (ShotNo, t_LGin[i_time], i_line+1)
                        #fig1.savefig(filepath +"/"+ filename, bbox_inches="tight", pad_inches=-0.033)
                        plt.close(fig1)
                    elif(self.mode == "ECHsetting_estimation"):
                        fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
                        ax1.axis("off")
                        ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False,
                                        labelleft=False,
                                        labelright=False, labeltop=False)
                        fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)

                        arr_input_2D[:8, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 2 + 4 * i_tank],
                                                        xmin=self.norm_fact[15][1], xmax=self.norm_fact[15][2])
                        arr_input_2D[8:16, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 3 + 4 * i_tank],
                                                          xmin=self.norm_fact[16][1], xmax=self.norm_fact[16][2])
                        arr_input_2D[16:24, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 4 + 4 * i_tank],
                                                           xmin=self.norm_fact[17][1], xmax=self.norm_fact[17][2])
                        arr_input_2D[24:32, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 5 + 4 * i_tank],
                                                           xmin=self.norm_fact[18][1], xmax=self.norm_fact[18][2])
                        max_arr_input_2D = np.max(arr_input_2D)
                        #if i_line == 2:
                        #    print("         line 3: ", np.max(arr_input_2D))
                        #else:
                        #    print("line %d: %.3f" % (i_line+1, np.max(arr_input_2D)))
                        arr_input_2D[32:40, :81] = Te[:81]
                        arr_input_2D[40:48, :80] = Te[81:]
                        arr_input_2D[48:56, :81] = ne[:81]
                        arr_input_2D[56:64, :80] = ne[81:]
                        arr_input_2D[65:70, :] = RAX  # RAX[m]
                        arr_input_2D[70:75, :] = Bq  # Bq[%]
                        arr_input_2D[75:80, :] = gamma  # gamma['']
                        arr_input_2D[80:85, :] = p0[idx_time_tsmap_nel]  # p0 [%]
                        arr_input_2D[85:90, :] = pf[idx_time_tsmap_nel]  # pf [arb]
                        arr_input_2D[90:95, :] = ip[idx_time_tsmap_nel]  # ip [kA/T]
                        arr_input_2D[95:, :] = ipf[idx_time_tsmap_nel]  # ipf [arb]
                        ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                        #filepath = "Cycle23rd/ECHsetting_est_Pmax2MW_line7/input"
                        filepath = "test/input"
                        filename = "dataset4GAN_LHDGAUSS_setting_est_SN%d_t%.3fs_ECHline%d.png" % (
                            ShotNo, t_LGin[i_time], i_line + 1)
                        fig1.savefig(filepath + "/" + filename, bbox_inches="tight", pad_inches=-0.000)#0.033)
                        plt.close(fig1)

                        arr_input_2D[:15, :] = normalize(data_3d_LGin[i_time, i_line, 3], xmin=self.norm_fact[9][1],
                                                         xmax=self.norm_fact[9][2])  # power [MW]
                        #if i_line == 2:
                        #    print("PW line 3: ", np.max(arr_input_2D[:15, :]))
                        arr_input_2D[15:32, :] = normalize(data_3d_LGin[i_time, i_line, 4], xmin=self.norm_fact[10][1],
                                                           xmax=self.norm_fact[10][2])  # Rfocus [m]
                        arr_input_2D[32:49, :] = normalize(data_3d_LGin[i_time, i_line, 5], xmin=self.norm_fact[11][1],
                                                           xmax=self.norm_fact[11][2])  # Tfocus [m]
                        arr_input_2D[49:66, :] = normalize(data_3d_LGin[i_time, i_line, 6], xmin=self.norm_fact[12][1],
                                                           xmax=self.norm_fact[12][2])  # Zfocus [m]
                        arr_input_2D[66:83, :] = normalize(data_3d_LGin[i_time, i_line, 13], xmin=self.norm_fact[13][1],
                                                           xmax=self.norm_fact[13][2])  # alpha [deg]
                        arr_input_2D[83:, :] = normalize(data_3d_LGin[i_time, i_line, 14], xmin=self.norm_fact[14][1],
                                                         xmax=self.norm_fact[14][2])  # beta [deg]

                        ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                        #filepath = "Cycle23rd/ECHsetting_est_Pmax2MW_line7/output"
                        filepath = "test/output"
                        filename = "dataset4GAN_LHDGAUSS_setting_est_SN%d_t%.3fs_ECHline%d.png" % (
                            ShotNo, t_LGin[i_time], i_line + 1)
                        fig1.savefig(filepath + "/" + filename, bbox_inches="tight", pad_inches=-0.000)#0.033)
                        plt.close(fig1)

                    else:
                        print("mode must be deposition_calculation or ECHsetting_estimation")
                        break
                else:
                    print("No data of LHDGAUSS at t=%.3f" % t_LGin[i_time])

    def make_dataset_line3(self, ShotNo):
        data_shotinfo = AnaData.retrieve('shotinfo', ShotNo, 1)
        RAX = float(data_shotinfo.getValData('Rax_vac'))
        Bq = float(data_shotinfo.getValData('Bq'))
        gamma = float(data_shotinfo.getValData('gamma'))

        data_tsmap_nel = AnaData.retrieve('tsmap_nel', ShotNo, 1)
        p0 = data_tsmap_nel.getValData('p0')
        pf = data_tsmap_nel.getValData('pf')
        ip = data_tsmap_nel.getValData('ip')
        ipf = data_tsmap_nel.getValData('ipf')
        data_LGdep = AnaData.retrieve('LHDGAUSS_DEPROF', ShotNo, 1)
        data_LGin = AnaData.retrieve('LHDGAUSS_INPUT', ShotNo, 1)
        data_TS = AnaData.retrieve('tsmap_smooth', ShotNo, 1)
        t_tsmap_nel = data_tsmap_nel.getDimData('Time')
        t_LGin = data_LGin.getDimData('time')
        t_LGdep = data_LGdep.getDimData('time')
        t_TS = data_TS.getDimData('Time')
        #reff = data.getValData('reff/a99')
        Te_fit = data_TS.getValData('Te_fit')
        ne_fit = data_TS.getValData('ne_fit')

        dsize_time = data_LGin.getDimSize(0)
        dsize_line = data_LGin.getDimSize(1)
        vnum_LGin = data_LGin.getValNo()
        data_3d_LGin = np.zeros((dsize_time, dsize_line, vnum_LGin))

        #normalization
        Te_fit = normalize(Te_fit, xmin=self.norm_fact[0][1], xmax=self.norm_fact[0][2])
        ne_fit = normalize(ne_fit, xmin=self.norm_fact[1][1], xmax=self.norm_fact[1][2])
        RAX = normalize(RAX , xmin=self.norm_fact[2][1], xmax=self.norm_fact[2][2])
        Bq = normalize(Bq, xmin=self.norm_fact[3][1], xmax=self.norm_fact[3][2])
        gamma = normalize(gamma, xmin=self.norm_fact[4][1], xmax=self.norm_fact[4][2])
        p0 = normalize(p0, xmin=self.norm_fact[5][1], xmax=self.norm_fact[5][2])
        pf = normalize(pf, xmin=self.norm_fact[6][1], xmax=self.norm_fact[6][2])
        ip = normalize(ip, xmin=self.norm_fact[7][1], xmax=self.norm_fact[7][2])
        ipf = normalize(ipf, xmin=self.norm_fact[8][1], xmax=self.norm_fact[8][2])
        for i in range(vnum_LGin):
            data_3d_LGin[:,:,i] = data_LGin.getValData(i)
        vnum_LGdep = data_LGdep.getValNo()
        data_3d_LGdep = np.zeros((data_LGdep.getDimSize(0), data_LGdep.getDimSize(1), vnum_LGdep))
        for i in range(vnum_LGdep):
            data_3d_LGdep[:,:,i] = data_LGdep.getValData(i)

        dpi = 100
        figsize_px = np.array([100, 100])
        figsize_inch = figsize_px / dpi
        for i_time in range(dsize_time):
            #for i_line in range(dsize_line):
            i_line = 2
            i_tank = 3
            if(data_3d_LGin[i_time, i_line, 3]>0.0):    #get data if Pech_inj > 0.0 MW
                arr_input_2D = np.zeros((100, 100))
                idx_time_TS = getNearestIndex(t_TS, t_LGin[i_time])
                idx_time_LGdep = getNearestIndex(t_LGdep, t_LGin[i_time])
                idx_time_tsmap_nel = getNearestIndex(t_tsmap_nel, t_LGin[i_time])
                if(t_LGin[i_time]==t_LGdep[idx_time_LGdep] and t_LGin[i_time]==t_TS[idx_time_TS]):
                    Te = Te_fit[idx_time_TS, :]
                    ne = ne_fit[idx_time_TS, :]
                    if(self.mode == "deposition_calculation"):
                        arr_input_2D[:8, :81] = Te[:81]
                        arr_input_2D[8:16, :80] = Te[81:]
                        arr_input_2D[16:24, :81] = ne[:81]
                        arr_input_2D[24:32, :80] = ne[81:]
                        arr_input_2D[35:40, :] = RAX    #RAX [m]
                        arr_input_2D[40:45, :] = Bq     #Bq[%]
                        arr_input_2D[45:50, :] = gamma  #gamma['']
                        arr_input_2D[50:55, :] = p0[idx_time_tsmap_nel]    #p0 [%]
                        arr_input_2D[55:60, :] = pf[idx_time_tsmap_nel]    #pf [arb]
                        arr_input_2D[60:65, :] = ip[idx_time_tsmap_nel]    #ip [kA/T]
                        arr_input_2D[65:70, :] = ipf[idx_time_tsmap_nel]    #ipf [arb]
                        arr_input_2D[70:75, :] = normalize(data_3d_LGin[i_time, i_line, 3], xmin=self.norm_fact[9][1], xmax=self.norm_fact[9][2])    #power [MW]
                        arr_input_2D[75:80, :] = normalize(data_3d_LGin[i_time, i_line+1, 4], xmin=self.norm_fact[10][1], xmax=self.norm_fact[10][2])    #Rfocus [m]
                        arr_input_2D[80:85, :] = normalize(data_3d_LGin[i_time, i_line+1, 5], xmin=self.norm_fact[11][1], xmax=self.norm_fact[11][2])    #Tfocus [m]
                        arr_input_2D[85:90, :] = normalize(data_3d_LGin[i_time, i_line+1, 6], xmin=self.norm_fact[12][1], xmax=self.norm_fact[12][2])    #Zfocus [m]
                        arr_input_2D[90:95, :] = normalize(data_3d_LGin[i_time, i_line+1, 13], xmin=self.norm_fact[13][1], xmax=self.norm_fact[13][2])   #alpha [deg]
                        arr_input_2D[95:, :] = normalize(data_3d_LGin[i_time, i_line+1, 14], xmin=self.norm_fact[14][1], xmax=self.norm_fact[14][2])     #beta [deg]

                        fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
                        ax1.axis("off")
                        ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False,
                                        labelleft=False,
                                        labelright=False, labeltop=False)
                        fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)
                        ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                        filepath = "ECHdeposition_est_Pmax2MW_line3/input"
                        filename = "dataset4GAN_LHDGAUSS_SN%d_t%.3fs_ECHline%d.png" % (ShotNo, t_LGin[i_time], i_line+1)
                        #fig1.savefig(filepath +"/"+ filename, bbox_inches="tight", pad_inches=-0.033)
                        plt.close(fig1)

                        arr_input_2D[:25,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 2+4*i_tank], xmin=self.norm_fact[15][1], xmax=self.norm_fact[15][2])
                        arr_input_2D[25:50,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 3+4*i_tank], xmin=self.norm_fact[16][1], xmax=self.norm_fact[16][2])
                        arr_input_2D[50:75,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 4+4*i_tank], xmin=self.norm_fact[17][1], xmax=self.norm_fact[17][2])
                        arr_input_2D[75:,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 5+4*i_tank], xmin=self.norm_fact[18][1], xmax=self.norm_fact[18][2])
                        ax1.imshow(arr_input_2D, vmin=0, vmax=1,interpolation='none', cmap='PuRd')
                        filepath = "ECHdeposition_est_Pmax2MW_line3/output"
                        filename = "dataset4GAN_LHDGAUSS_SN%d_t%.3fs_ECHline%d.png" % (ShotNo, t_LGin[i_time], i_line+1)
                        #fig1.savefig(filepath +"/"+ filename, bbox_inches="tight", pad_inches=-0.033)
                        plt.close(fig1)
                    elif(self.mode == "ECHsetting_estimation"):
                        fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
                        ax1.axis("off")
                        ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False,
                                        labelleft=False,
                                        labelright=False, labeltop=False)
                        fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)

                        arr_input_2D[:8, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 2 + 4 * i_tank],
                                                        xmin=self.norm_fact[15][1], xmax=self.norm_fact[15][2])
                        arr_input_2D[8:16, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 3 + 4 * i_tank],
                                                          xmin=self.norm_fact[16][1], xmax=self.norm_fact[16][2])
                        arr_input_2D[16:24, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 4 + 4 * i_tank],
                                                           xmin=self.norm_fact[17][1], xmax=self.norm_fact[17][2])
                        arr_input_2D[24:32, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 5 + 4 * i_tank],
                                                           xmin=self.norm_fact[18][1], xmax=self.norm_fact[18][2])
                        max_arr_input_2D = np.max(arr_input_2D)
                        #if i_line == 2:
                        #    print("         line 3: ", np.max(arr_input_2D))
                        #else:
                        #    print("line %d: %.3f" % (i_line+1, np.max(arr_input_2D)))
                        arr_input_2D[32:40, :81] = Te[:81]
                        arr_input_2D[40:48, :80] = Te[81:]
                        arr_input_2D[48:56, :81] = ne[:81]
                        arr_input_2D[56:64, :80] = ne[81:]
                        arr_input_2D[65:70, :] = RAX  # RAX[m]
                        arr_input_2D[70:75, :] = Bq  # Bq[%]
                        arr_input_2D[75:80, :] = gamma  # gamma['']
                        arr_input_2D[80:85, :] = p0[idx_time_tsmap_nel]  # p0 [%]
                        arr_input_2D[85:90, :] = pf[idx_time_tsmap_nel]  # pf [arb]
                        arr_input_2D[90:95, :] = ip[idx_time_tsmap_nel]  # ip [kA/T]
                        arr_input_2D[95:, :] = ipf[idx_time_tsmap_nel]  # ipf [arb]
                        ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                        filepath = "ECHsetting_est_Pmax2MW_line3/input"
                        filename = "dataset4GAN_LHDGAUSS_setting_est_SN%d_t%.3fs_ECHline%d.png" % (
                            ShotNo, t_LGin[i_time], i_line + 1)
                        fig1.savefig(filepath + "/" + filename, bbox_inches="tight", pad_inches=-0.033)
                        plt.close(fig1)

                        arr_input_2D[:15, :] = normalize(data_3d_LGin[i_time, i_line, 3], xmin=self.norm_fact[9][1],
                                                         xmax=self.norm_fact[9][2])  # power [MW]
                        #if i_line == 2:
                        #    print("PW line 3: ", np.max(arr_input_2D[:15, :]))
                        arr_input_2D[15:32, :] = normalize(data_3d_LGin[i_time, i_line+1, 4], xmin=self.norm_fact[10][1],
                                                           xmax=self.norm_fact[10][2])  # Rfocus [m]
                        arr_input_2D[32:49, :] = normalize(data_3d_LGin[i_time, i_line+1, 5], xmin=self.norm_fact[11][1],
                                                           xmax=self.norm_fact[11][2])  # Tfocus [m]
                        arr_input_2D[49:66, :] = normalize(data_3d_LGin[i_time, i_line+1, 6], xmin=self.norm_fact[12][1],
                                                           xmax=self.norm_fact[12][2])  # Zfocus [m]
                        arr_input_2D[66:83, :] = normalize(data_3d_LGin[i_time, i_line+1, 13], xmin=self.norm_fact[13][1],
                                                           xmax=self.norm_fact[13][2])  # alpha [deg]
                        arr_input_2D[83:, :] = normalize(data_3d_LGin[i_time, i_line+1, 14], xmin=self.norm_fact[14][1],
                                                         xmax=self.norm_fact[14][2])  # beta [deg]

                        ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                        filepath = "ECHsetting_est_Pmax2MW_line3/output"
                        filename = "dataset4GAN_LHDGAUSS_setting_est_SN%d_t%.3fs_ECHline%d.png" % (
                            ShotNo, t_LGin[i_time], i_line + 1)
                        fig1.savefig(filepath + "/" + filename, bbox_inches="tight", pad_inches=-0.033)
                        plt.close(fig1)

                    else:
                        print("mode must be deposition_calculation or ECHsetting_estimation")
                        break
                else:
                    print("No data of LHDGAUSS at t=%.3f" % t_LGin[i_time])
    def make_dataset(self, ShotNo):
        data_shotinfo = AnaData.retrieve('shotinfo', ShotNo, 1)
        RAX = float(data_shotinfo.getValData('Rax_vac'))
        Bq = float(data_shotinfo.getValData('Bq'))
        gamma = float(data_shotinfo.getValData('gamma'))

        data_tsmap_nel = AnaData.retrieve('tsmap_nel', ShotNo, 1)
        p0 = data_tsmap_nel.getValData('p0')
        pf = data_tsmap_nel.getValData('pf')
        ip = data_tsmap_nel.getValData('ip')
        ipf = data_tsmap_nel.getValData('ipf')
        data_LGdep = AnaData.retrieve('LHDGAUSS_DEPROF', ShotNo, 1)
        data_LGin = AnaData.retrieve('LHDGAUSS_INPUT', ShotNo, 1)
        data_TS = AnaData.retrieve('tsmap_smooth', ShotNo, 1)
        t_tsmap_nel = data_tsmap_nel.getDimData('Time')
        t_LGin = data_LGin.getDimData('time')
        t_LGdep = data_LGdep.getDimData('time')
        t_TS = data_TS.getDimData('Time')
        #reff = data.getValData('reff/a99')
        Te_fit = data_TS.getValData('Te_fit')
        ne_fit = data_TS.getValData('ne_fit')

        dsize_time = data_LGin.getDimSize(0)
        dsize_line = data_LGin.getDimSize(1)
        vnum_LGin = data_LGin.getValNo()
        data_3d_LGin = np.zeros((dsize_time, dsize_line, vnum_LGin))

        #normalization
        Te_fit = normalize(Te_fit, xmin=self.norm_fact[0][1], xmax=self.norm_fact[0][2])
        ne_fit = normalize(ne_fit, xmin=self.norm_fact[1][1], xmax=self.norm_fact[1][2])
        RAX = normalize(RAX , xmin=self.norm_fact[2][1], xmax=self.norm_fact[2][2])
        Bq = normalize(Bq, xmin=self.norm_fact[3][1], xmax=self.norm_fact[3][2])
        gamma = normalize(gamma, xmin=self.norm_fact[4][1], xmax=self.norm_fact[4][2])
        p0 = normalize(p0, xmin=self.norm_fact[5][1], xmax=self.norm_fact[5][2])
        pf = normalize(pf, xmin=self.norm_fact[6][1], xmax=self.norm_fact[6][2])
        ip = normalize(ip, xmin=self.norm_fact[7][1], xmax=self.norm_fact[7][2])
        ipf = normalize(ipf, xmin=self.norm_fact[8][1], xmax=self.norm_fact[8][2])
        for i in range(vnum_LGin):
            data_3d_LGin[:,:,i] = data_LGin.getValData(i)
        vnum_LGdep = data_LGdep.getValNo()
        data_3d_LGdep = np.zeros((data_LGdep.getDimSize(0), data_LGdep.getDimSize(1), vnum_LGdep))
        for i in range(vnum_LGdep):
            data_3d_LGdep[:,:,i] = data_LGdep.getValData(i)

        dpi = 100
        figsize_px = np.array([100, 100])
        figsize_inch = figsize_px / dpi
        for i_time in range(dsize_time):
            for i_line in range(dsize_line):
                if(data_3d_LGin[i_time, i_line, 3]>0.0):    #get data if Pech_inj > 0.0 MW
                    arr_input_2D = np.zeros((100, 100))
                    idx_time_TS = getNearestIndex(t_TS, t_LGin[i_time])
                    idx_time_LGdep = getNearestIndex(t_LGdep, t_LGin[i_time])
                    idx_time_tsmap_nel = getNearestIndex(t_tsmap_nel, t_LGin[i_time])
                    if(t_LGin[i_time]==t_LGdep[idx_time_LGdep] and t_LGin[i_time]==t_TS[idx_time_TS]):
                        Te = Te_fit[idx_time_TS, :]
                        ne = ne_fit[idx_time_TS, :]
                        if(self.mode == "deposition_calculation"):
                            arr_input_2D[:8, :81] = Te[:81]
                            arr_input_2D[8:16, :80] = Te[81:]
                            arr_input_2D[16:24, :81] = ne[:81]
                            arr_input_2D[24:32, :80] = ne[81:]
                            arr_input_2D[35:40, :] = RAX    #RAX [m]
                            arr_input_2D[40:45, :] = Bq     #Bq[%]
                            arr_input_2D[45:50, :] = gamma  #gamma['']
                            arr_input_2D[50:55, :] = p0[idx_time_tsmap_nel]    #p0 [%]
                            arr_input_2D[55:60, :] = pf[idx_time_tsmap_nel]    #pf [arb]
                            arr_input_2D[60:65, :] = ip[idx_time_tsmap_nel]    #ip [kA/T]
                            arr_input_2D[65:70, :] = ipf[idx_time_tsmap_nel]    #ipf [arb]
                            arr_input_2D[70:75, :] = normalize(data_3d_LGin[i_time, i_line, 3], xmin=self.norm_fact[9][1], xmax=self.norm_fact[9][2])    #power [MW]
                            arr_input_2D[75:80, :] = normalize(data_3d_LGin[i_time, i_line, 4], xmin=self.norm_fact[10][1], xmax=self.norm_fact[10][2])    #Rfocus [m]
                            arr_input_2D[80:85, :] = normalize(data_3d_LGin[i_time, i_line, 5], xmin=self.norm_fact[11][1], xmax=self.norm_fact[11][2])    #Tfocus [m]
                            arr_input_2D[85:90, :] = normalize(data_3d_LGin[i_time, i_line, 6], xmin=self.norm_fact[12][1], xmax=self.norm_fact[12][2])    #Zfocus [m]
                            arr_input_2D[90:95, :] = normalize(data_3d_LGin[i_time, i_line, 13], xmin=self.norm_fact[13][1], xmax=self.norm_fact[13][2])   #alpha [deg]
                            arr_input_2D[95:, :] = normalize(data_3d_LGin[i_time, i_line, 14], xmin=self.norm_fact[14][1], xmax=self.norm_fact[14][2])     #beta [deg]

                            fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
                            ax1.axis("off")
                            ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False,
                                            labelleft=False,
                                            labelright=False, labeltop=False)
                            fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)
                            ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                            filepath = "ECHdeposition_est_Pmax2MW/input"
                            filename = "dataset4GAN_LHDGAUSS_SN%d_t%.3fs_ECHline%d.png" % (ShotNo, t_LGin[i_time], i_line+1)
                            fig1.savefig(filepath +"/"+ filename, bbox_inches="tight", pad_inches=-0.033)
                            plt.close(fig1)

                            arr_input_2D[:25,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 2+4*i_line], xmin=self.norm_fact[15][1], xmax=self.norm_fact[15][2])
                            arr_input_2D[25:50,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 3+4*i_line], xmin=self.norm_fact[16][1], xmax=self.norm_fact[16][2])
                            arr_input_2D[50:75,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 4+4*i_line], xmin=self.norm_fact[17][1], xmax=self.norm_fact[17][2])
                            arr_input_2D[75:,:] = normalize(data_3d_LGdep[idx_time_LGdep, :, 5+4*i_line], xmin=self.norm_fact[18][1], xmax=self.norm_fact[18][2])
                            ax1.imshow(arr_input_2D, vmin=0, vmax=1,interpolation='none', cmap='PuRd')
                            filepath = "ECHdeposition_est_Pmax2MW/output"
                            filename = "dataset4GAN_LHDGAUSS_SN%d_t%.3fs_ECHline%d.png" % (ShotNo, t_LGin[i_time], i_line+1)
                            #fig1.savefig(filepath +"/"+ filename, bbox_inches="tight", pad_inches=-0.033)
                            plt.close(fig1)
                        elif(self.mode == "ECHsetting_estimation"):
                            fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
                            ax1.axis("off")
                            ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False,
                                            labelleft=False,
                                            labelright=False, labeltop=False)
                            fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)

                            arr_input_2D[:8, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 2 + 4 * i_line],
                                                             xmin=self.norm_fact[15][1], xmax=self.norm_fact[15][2])
                            arr_input_2D[8:16, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 3 + 4 * i_line],
                                                               xmin=self.norm_fact[16][1], xmax=self.norm_fact[16][2])
                            arr_input_2D[16:24, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 4 + 4 * i_line],
                                                               xmin=self.norm_fact[17][1], xmax=self.norm_fact[17][2])
                            arr_input_2D[24:32, :] = normalize(data_3d_LGdep[idx_time_LGdep, :, 5 + 4 * i_line],
                                                             xmin=self.norm_fact[18][1], xmax=self.norm_fact[18][2])
                            max_arr_input_2D = np.max(arr_input_2D)
                            if i_line == 3:
                                print("         line 3: ", np.max(arr_input_2D))
                            #else:
                            #    print("line %d: %.3f" % (i_line+1, np.max(arr_input_2D)))
                            arr_input_2D[32:40, :81] = Te[:81]
                            arr_input_2D[40:48, :80] = Te[81:]
                            arr_input_2D[48:56, :81] = ne[:81]
                            arr_input_2D[56:64, :80] = ne[81:]
                            arr_input_2D[65:70, :] = RAX  # RAX[m]
                            arr_input_2D[70:75, :] = Bq  # Bq[%]
                            arr_input_2D[75:80, :] = gamma  # gamma['']
                            arr_input_2D[80:85, :] = p0[idx_time_tsmap_nel]  # p0 [%]
                            arr_input_2D[85:90, :] = pf[idx_time_tsmap_nel]  # pf [arb]
                            arr_input_2D[90:95, :] = ip[idx_time_tsmap_nel]  # ip [kA/T]
                            arr_input_2D[95:, :] = ipf[idx_time_tsmap_nel]  # ipf [arb]
                            ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                            filepath = "ECHsetting_est_Pmax2MW/input"
                            filename = "dataset4GAN_LHDGAUSS_setting_est_SN%d_t%.3fs_ECHline%d.png" % (
                            ShotNo, t_LGin[i_time], i_line + 1)
                            fig1.savefig(filepath + "/" + filename, bbox_inches="tight", pad_inches=-0.033)
                            plt.close(fig1)

                            arr_input_2D[:15, :] = normalize(data_3d_LGin[i_time, i_line, 3], xmin=self.norm_fact[9][1],
                                                             xmax=self.norm_fact[9][2])  # power [MW]
                            if i_line == 3:
                                print("PW line 3: ", np.max(arr_input_2D[:15, :]))
                            arr_input_2D[15:32, :] = normalize(data_3d_LGin[i_time, i_line, 4], xmin=self.norm_fact[10][1],
                                                               xmax=self.norm_fact[10][2])  # Rfocus [m]
                            arr_input_2D[32:49, :] = normalize(data_3d_LGin[i_time, i_line, 5], xmin=self.norm_fact[11][1],
                                                               xmax=self.norm_fact[11][2])  # Tfocus [m]
                            arr_input_2D[49:66, :] = normalize(data_3d_LGin[i_time, i_line, 6], xmin=self.norm_fact[12][1],
                                                               xmax=self.norm_fact[12][2])  # Zfocus [m]
                            arr_input_2D[66:83, :] = normalize(data_3d_LGin[i_time, i_line, 13], xmin=self.norm_fact[13][1],
                                                               xmax=self.norm_fact[13][2])  # alpha [deg]
                            arr_input_2D[83:, :] = normalize(data_3d_LGin[i_time, i_line, 14], xmin=self.norm_fact[14][1],
                                                             xmax=self.norm_fact[14][2])  # beta [deg]

                            ax1.imshow(arr_input_2D, vmin=0, vmax=1, interpolation='none', cmap='PuRd')
                            filepath = "ECHsetting_est_Pmax2MW/output"
                            filename = "dataset4GAN_LHDGAUSS_setting_est_SN%d_t%.3fs_ECHline%d.png" % (
                                ShotNo, t_LGin[i_time], i_line + 1)
                            #fig1.savefig(filepath + "/" + filename, bbox_inches="tight", pad_inches=-0.033)
                            plt.close(fig1)

                        else:
                            print("mode must be deposition_calculation or ECHsetting_estimation")
                            break
                    else:
                        print("No data of LHDGAUSS at t=%.3f" % t_LGin[i_time])

    def get_dataseries(self):
        #for i in range(171001, 172000):#Cycle 23: 170017-178979):	#Cycle 20: 144105-153366
        for i in tqdm.tqdm(range(178001, 178980)):  # Cycle 23: 170017-178979):	#Cycle 20: 144105-153366
            try:
                if icheckfile('LHDGAUSS_DEPROF', i, 1) & icheckfile('LHDGAUSS_INPUT', i, 1):
                    print("Data exist for ShotNo.%d" % i)
                    md4GAN.make_dataset_line7(i)
                else:
                    print("Data NOT exist or BROKEN for ShotNo.%d" % i)
            except Exception as e:
                print("args:", e.args)
                print("Data NOT exist for ShotNo.%d" % i)

    def searchData(self):
        for i in range(156277, 154000):
            try:
                if icheckfile('LHDGAUSS_DEPROF', i, 1) & icheckfile('LHDGAUSS_INPUT', i, 1):
                #if AnaData.retrieve('LHDGAUSS_INPUT', i, 1):
                    #test1 = icheckfile('LHDGAUSS_DEPROF', i, 1)
                    #test2 = icheckfile('LHDGAUSS_INPUT', i, 1)
                    data = AnaData.retrieve('LHDGAUSS_INPUT', i, 1)
                    data = AnaData.retrieve('LHDGAUSS_DEPROF', i, 1)
                    data = AnaData.retrieve('shotinfo', i, 1)
                    data = AnaData.retrieve('tsmap_smooth', i, 1)
                    data = AnaData.retrieve('tsmap_nel', i, 1)
                    print("Data exist for ShotNo.%d" % i)
                else:
                    print("Data NOT exist for ShotNo.%d" % i)
            #except ZeroDivisionError:
            #    pass
            except Exception as e:
                print("例外args:", e.args)
                print("Data NOT exist for ShotNo.%d" % i)

    def plot_tsmap_LHDGAUSS_input(self, val_time):
        arr_input_2D = np.zeros((100,100))
        data_3d = self.ana_plot_LHDGauss_input(val_time)
        reff, Te_fit, ne_fit = self.get_tsmap(val_time)
        #plt.plot(reff, Te_fit)
        #plt.show()
        arr_input_2D[:12,:71] = Te_fit[-71:]
        arr_input_2D[12:24,:71] = ne_fit[-71:]
        buf_arr = np.ones((100, 19))
        arr_input_2D[24::4,:] = (buf_arr*data_3d[0,3:]).T
        arr_input_2D[25::4,:] = (buf_arr*data_3d[0,3:]).T
        arr_input_2D[26::4,:] = (buf_arr*data_3d[0,3:]).T
        arr_input_2D[27::4,:] = (buf_arr*data_3d[0,3:]).T
        #plt.imshow(data_3d)
        #plt.show()
        #plt.imshow(arr_input_2D, vmin=0, vmax=2,interpolation='none')
        #plt.show()
        dpi = 100
        figsize_px = np.array([100,100])
        figsize_inch = figsize_px/dpi
        fig1, ax1 = plt.subplots(figsize=figsize_inch, dpi=dpi)
        ax1.axis("off")
        ax1.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)
        fig1.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        ax1.imshow(arr_input_2D, vmin=0, vmax=2,interpolation='none', cmap='jet')
        fig1.savefig('tsmap_LHDGAUSS_input_test.png', bbox_inches="tight", pad_inches=-0.033)
        plt.close()

    def ana_plot_LHDGauss_input(self, val_time):
        data = AnaData.retrieve('LHDGAUSS_INPUT', self.ShotNo, 1)
        print(data.getDiagnostics())
        print(data.getComment())
        print(data.getDate())
        print(data.getShotNo())

        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))

        t = data.getDimData('time')
        l = data.getDimData('line#')

        data_3d = np.zeros((data.getDimSize(0), data.getDimSize(1), vnum))

        for i in range(vnum):
            data_3d[:,:,i] = data.getValData(i)

        '''
        power = data.getValData('power')
        rfocus = data.getValData('Rfocus')
        tfocus = data.getValData('Tfocus')
        zfocus = data.getValData('Zfocus')
        xstart = data.getValData('xstart')
        ystart = data.getValData('xstart')
        zstart = data.getValData('xstart')
        xtarget = data.getValData('xtarget')
        ytarget = data.getValData('ytarget')
        ztarget = data.getValData('ztarget')
        alpha = data.getValData('alpha')
        beta = data.getValData('beta')
        enxx = data.getValData('enxx')
        enxy = data.getValData('enxy')
        enxz = data.getValData('enxz')
        w0x = data.getValData('w0x')
        w0y = data.getValData('w0y')
        z0x = data.getValData('z0x')
        z0y = data.getValData('z0y')
        '''

        idx_time = getNearestIndex(t, val_time)
        #plt.title('LHDGAUSS_INPUT, t=%.3fsec' % t[idx_time])
        #plt.xlabel('')
        #plt.ylabel('')
        #plt.imshow(data_3d[idx_time, :, :], vmax=5, cmap='jet')
        #plt.show()  # グラフ描画

        return data_3d[idx_time, :, :]

    def ana_plot_LHDGauss_deprof_3D(self, val_time):
        data = AnaData.retrieve('LHDGAUSS_DEPROF', self.ShotNo, 1)
        print(data.getDiagnostics())
        print(data.getComment())
        print(data.getDate())
        print(data.getShotNo())

        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        reff = data.getDimData('reff')

        t = data.getDimData('time')

        data_3d = np.zeros((data.getDimSize(0), data.getDimSize(1), vnum))

        for i in range(vnum):
            data_3d[:,:,i] = data.getValData(i)

        idx_time = getNearestIndex(t, val_time)

        plt.title('LHDGAUSS_DEPROF, t=%.3fsec' % t[idx_time])
        plt.xlabel('')
        plt.ylabel('')
        labels = []

        plt.imshow(data_3d[:, :, 0].T, cmap='jet')
        plt.show()

        i_line = 3
        ch = 3
        #plt.imshow(data_3d[idx_time, :, :].T, cmap='jet')
        #plt.imshow(data_3d[idx_time, :, 1+4*ch:1+4*(ch+1)].T, cmap='jet', aspect=80)
        plt.clf()
        plt.figure(figsize=(4,3), dpi=150)
        plt.plot(reff, data_3d[idx_time, :, 2 + 4*i_line], label='O_Power_Density')
        plt.plot(reff, data_3d[idx_time, :, 3 + 4*i_line], label='O_Total_Power')
        plt.plot(reff, data_3d[idx_time, :, 4 + 4*i_line], label='X_Power_Density')
        plt.plot(reff, data_3d[idx_time, :, 5 + 4*i_line], label='X_Total_Power')
        plt.legend()  # 凡例セット
        label = 't = %05.3f (sec)' % (t[idx_time])  # 凡例用文字列
        title = ('LHDGAUSS_DEPROF \n#%d, ' + label) % (self.ShotNo)
        plt.title(title, loc='right')
        plt.ylim(0, 5)
        plt.xlim(0, 1)
        plt.xlabel("reff")
        plt.ylabel("Power Density [MW/m3],\n Total Power [MW]")
        plt.tight_layout()
        plt.savefig('LHDGauss_deprof_SN' + str(self.ShotNo) + '_t' + str(t[idx_time]) + '.png')
        #plt.show()  # グラフ描画



        arr_input_2D = np.zeros((100,100))
        #plt.figure(figsize=(20, 50))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '3%', pad=-2.2)
        im = ax.imshow(data_3d[:,::-1, 4 + 4 * i_line].T, extent=(np.min(t),np.max(t) , 0, 1), aspect=2)
        ax.set_title(self.ShotNo, loc='right')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('reff')
        #plt.xlabel('Time [sec]')
        #plt.ylabel('reff')
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Power Density [MW/m3]')
        #plt.show()
        plt.savefig('LHDGauss_deprof_3D_SN' + str(self.ShotNo) + '.png')
        arr_input_2D[:25,:] = data_3d[idx_time, :, 2 + 4 * i_line]
        arr_input_2D[25:50,:] = data_3d[idx_time, :, 3 + 4 * i_line]
        arr_input_2D[50:75,:] = data_3d[idx_time, :, 4 + 4 * i_line]
        arr_input_2D[75:,:] = data_3d[idx_time, :, 5 + 4 * i_line]
        #plt.show()
        dpi = 100
        figsize_px = np.array([100,100])
        figsize_inch = figsize_px/dpi
        fig, ax = plt.subplots(figsize=figsize_inch, dpi=dpi)
        ax.axis("off")
        ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                        labelright=False, labeltop=False)
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        ax.imshow(arr_input_2D, vmin=0, vmax=1,interpolation='none', cmap='PuRd')
        plt.show()
        #fig.savefig('LHDGAUSS_DEPROF_test.png', bbox_inches="tight", pad_inches=-0.033)

    def ana_plot_LHDGauss_deprof(self, arr_time):
        data = AnaData.retrieve('LHDGAUSS_DEPROF', self.ShotNo, 1)

        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        reff = data.getDimData('reff')

        t = data.getDimData('time')

        sum_power_density = data.getValData('Sum_Power_Density')
        sum_total_power = data.getValData('Sum_Total_Power')
        L07O_power_density = data.getValData('L07O_Power_Density')
        L07O_total_power = data.getValData('L07O_Total_Power')
        L07X_power_density = data.getValData('L07X_Power_Density')
        L07X_total_power = data.getValData('L07X_Total_Power')

        plt.figure(figsize=(5.3, 3), dpi=100)
        plt.xlabel('reff')
        plt.ylabel('Power Density [MW/m3], Total Power [MW]')
        labels = []

        idx_arr_time = getNearestIndex(t, arr_time)

        for i, idx in enumerate(idx_arr_time):
            sum_pd = sum_power_density[idx, :]
            sum_tp = sum_total_power[idx, :]
            l7o_pd = L07O_power_density[idx, :]
            l7o_tp = L07O_total_power[idx, :]
            l7x_pd = L07X_power_density[idx, :]
            l7x_tp = L07X_total_power[idx, :]
            #plt.plot(r, T)
            #plt.plot(r, n)
            plt.plot(reff, l7o_pd, label='L07O_Power_Density')
            plt.plot(reff, l7o_tp, label='L07O_Total_Power')
            plt.plot(reff, l7x_pd, label='L07X_Power_Density')
            plt.plot(reff, l7x_tp, label='L07X_Total_Power')

        plt.legend()  # 凡例セット
        label = 't = %05.3f (sec)' % (t[idx])  # 凡例用文字列
        title = ('LHDGAUSS_DEPROF #%d, ' + label) % (self.ShotNo)
        plt.title(title)
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        plt.show()  # グラフ描画

    def get_tsmap(self, val_time):
        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)

        #vnum = data.getValNo()
        #for i in range(vnum):
        #    print(i, data.getValName(i), data.getValUnit(i))

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')

        idx_time = getNearestIndex(t, val_time)

        return reff[idx_time, :], Te_fit[idx_time, :], ne_fit[idx_time, :]

    def ana_plot_timetrace_TS(self):
        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('reff')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        fig = plt.figure(figsize=(6, 2), dpi=150)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        plt.title(self.ShotNo, loc='right')
        #plt.xlim(3, 6)
        Te = data.getValData(0)
        ne = data.getValData(1)
        ln1 = ax1.plot(t, ne[:, 0], 'C0', label='ne0')
        ln2 = ax2.plot(t, Te[:, 0], 'C1', label='Te0')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='lower right')

        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('ne0[e19m-3]')
        ax2.set_ylabel('Te0[keV]')
        ax1.set_xlim(3.316, 9.983)
        ax1.set_ylim(0, 2)
        ax2.set_ylim(0, 5)
        #plt.legend()
        plt.savefig('tsmap_Te0_ne0_twinx_SN' + str(self.ShotNo) + '.png')
        #plt.show()

    def ana_plot_timetrace(self, arr_time):
        #data = AnaData.retrieve('fir_nel', self.ShotNo, 1)
        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        plt.figure(figsize=(32, 18), dpi=100)
        plt.xlabel('Time')
        plt.ylabel('ne_bar')
        plt.title(self.ShotNo)
        #plt.xlim(3, 6)
        for i in range(vnum):
            ne_bar = data.getValData(i)
            plt.plot(t, ne_bar)
        plt.show()

    def ana_plot_ece(self, arr_time):
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
        fig = plt.figure(figsize=(16, 18), dpi=100)
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)

        ax1.set_title(self.ShotNo, fontsize=18)
        ax1.set_ylabel('Te', fontsize=18)
        ax1.set_xlim(3, 6)
        ax2.set_xlabel('time', fontsize=18)
        ax2.set_ylabel('ne', fontsize=18)
        ax2.set_xlim(3, 6)
        for i in range(vnum):
            ax1.plot(t, Te)
            ne_bar = data.getValData(i)
            ax2.plot(t_fir, ne_bar)
        #plt.show()
        plt.savefig("timetrace_ECE_FIR_No%d.png" % self.ShotNo)


        '''
        # 各時刻毎の R 対 Te のデータをプロット

        plt.figure(figsize=(5.3, 3), dpi=100)
        plt.xlabel('R')
        plt.ylabel('Te [keV]')
        labels = []

        idx_arr_time = getNearestIndex(t, arr_time)
        val_arr_time = t[idx_arr_time]

        # データ取得
        T = Te[idx_arr_time, :]  # R 次元でのスライスを取得
        plt.plot(r, T, label='Te', linestyle='None', marker='o')

        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('reff')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        r_high = data.getValData('R_high')
        r_low = data.getValData('R_low')
        #ne = data.getValData('ne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')
        #label = 't = %05.3f (sec)' % (t[idx])  # 凡例用文字列
        #title = ('tsmap_smooth #%d, ' + label) % (self.ShotNo)
        #plt.title(title)
        idx_arr_time = getNearestIndex(t, arr_time)
        val_arr_time = t[idx_arr_time]

        # データ取得
        T = Te_fit[idx_arr_time, :]  # R 次元でのスライスを取得
        ne = ne_fit[idx_arr_time, :]  # R 次元でのスライスを取得
        R_ = r_low[idx_arr_time, :]  # R 次元でのスライスを取得
        plt.plot(R_, T, label='Te', linestyle='None', marker='o')
        plt.legend()  # 凡例セット
        plt.show()  # グラフ描画
        plt.plot(R_, ne, label='ne', linestyle='None', marker='o')
        #plt.ylim(0, 5)
        #plt.xlim(0, 1.5)

        data = AnaData.retrieve('fir_nel', self.ShotNo, 1)
        vnum = data.getValNo()

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        #r = data.getDimData('reff')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')
        idx_arr_time = getNearestIndex(t, arr_time)
        val_arr_time = t[idx_arr_time]

        ne_bar_prof = np.zeros(vnum)

        for i in range(vnum):
            ne_bar = data.getValData(i)
            ne_bar_prof[i] = ne_bar[idx_arr_time]
        plt.plot(self.R_fir, ne_bar_prof, linestyle='None', marker='v')
        plt.show()  # グラフ描画
        '''

    def ana_plot_eg(self, arr_time):
        data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('reff')
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #r_unit = data.getDimUnit('m')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        reff = data.getValData('reff/a99')
        #Te = data.getValData('Te')
        #dTe = data.getValData('dTe')
        #ne = data.getValData('ne_calFIR')
        #dne = data.getValData('dne_calFIR')
        Te_fit = data.getValData('Te_fit')
        ne_fit = data.getValData('ne_fit')

        # 各時刻毎の R 対 Te のデータをプロット

        plt.figure(figsize=(5.3, 3), dpi=100)
        plt.xlabel('reff/a99')
        plt.ylabel('Te [keV], ne [e19m-3]')
        labels = []

        idx_arr_time = getNearestIndex(t, arr_time)
        val_arr_time = t[idx_arr_time]

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
            plt.plot(r, T_fit, label='Te')
            plt.plot(r, n_fit, label='ne')

        label = 't = %05.3f (sec)' % (t[idx])  # 凡例用文字列
        title = ('tsmap_smooth #%d, ' + label) % (self.ShotNo)
        plt.title(title)
        plt.legend()  # 凡例セット
        plt.ylim(0, 5)
        plt.xlim(0, 1.5)

        plt.show()  # グラフ描画

    def ana_plot_tsmap_3D(self, val_time):
        data = AnaData.retrieve('tsmap_calib', self.ShotNo, 1)
        #data = AnaData.retrieve('tsmap_smooth', self.ShotNo, 1)
        print(data.getDiagnostics())
        print(data.getComment())
        print(data.getDate())
        print(data.getShotNo())

        dnum = data.getDimNo()
        for i in range(dnum):
            print(i, data.getDimName(i), data.getDimUnit(i))
        vnum = data.getValNo()
        for i in range(vnum):
            print(i, data.getValName(i), data.getValUnit(i))
        #reff = data.getDimData('reff')

        t = data.getDimData('Time')

        data_3d = np.zeros((data.getDimSize(0), data.getDimSize(1), vnum))

        for i in range(vnum):
            data_3d[:,:,i] = data.getValData(i)

        idx_time = getNearestIndex(t, val_time)
        plt.title('tsmap_calib, t=%.3fsec' % t[idx_time])
        plt.xlabel('')
        plt.ylabel('')
        labels = []


        #plt.imshow(data_3d[idx_time, :, :].T, cmap='jet')
        plt.imshow(data_3d[:, :, 8].T, cmap='jet', vmax=10)
        #plt.ylim(-0.5,3.5)
        #plt.xlim(0,99)
        plt.show()
        plt.imshow(data_3d[:, :, 12].T, cmap='jet', vmax=10)
        #plt.imshow(data_3d[:, :, 0].T, cmap='jet', vmax=10)
        plt.show()

    def ana_plot_TS(self):
        # トムソン のショット番号 8000, サブショット番号 1 のデータを取得 (クラス nifs.AnaData のインスタンスを返す)
        data = AnaData.retrieve('thomson', self.ShotNo, 1)

        # 次元 'R' のデータを取得 (要素数137 の一次元配列(numpy.Array)が返ってくる)
        r = data.getDimData('R')
        r_unit = data.getDimUnit('R')

        # 次元 'Time' のデータを取得 (要素数96 の一次元配列(numpy.Array)が返ってくる)
        t = data.getDimData('Time')

        # 変数 'Te' のデータを取得 (要素数 136x96 の二次元配列(numpy.Array)が返ってくる)
        Te = data.getValData('Te')

        # 各時刻毎の R 対 Te のデータをプロット

        plt.title('Thomson')
        plt.xlabel('R [' + r_unit + ']')
        plt.ylabel('Te')
        labels = []

        # 5 データ取得
        for i in range(200,205):
            T = Te[i, :]  # R 次元でのスライスを取得
            plt.plot(r, T)
            label = 't = %05.3f (sec)' % (t[i] / 1000.0)  # 凡例用文字列
            labels.append(label)

        plt.legend(labels)  # 凡例セット

        plt.show()  # グラフ描画

def normalize(x, xmin, xmax, amin=0, amax=1):
    """
    Normalize x between 0 and 1
    :param x:
    :param xmin:
    :param xmax:
    :param amin:
    :param amax:
    :return:
    """

    return (amax - amin) * (x - xmin) / (xmax - xmin) + amin



def getNearestIndex(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    return np.abs(np.asarray(list) - num).argmin()

def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]

def ftpList(targetpath):
    #print 'In ftpList %s' % targetpath
    ftp = FTP('egftp1.lhd.nifs.ac.jp')
    ftp.login()
    flist = ftp.nlst(targetpath)
    ftp.quit()
    return flist

def ftpSize(targetpath):
    #print 'In ftpList %s' % targetpath
    ftp = FTP('egftp1.lhd.nifs.ac.jp')
    ftp.login()
    flist_buf = ftp.nlst(targetpath)
    flist = str(flist_buf)[2:-2]
    size = ftp.size(flist)
    ftp.quit()
    return size

def icheckfile(diagname, shotno, subshot):
    targetfolderpath = 'data/'
    targetfolderpath = targetfolderpath + diagname + '/'
    firstshotno = int(shotno / 1000) * 1000
    firstfolder = '%06d/' % firstshotno
    targetfolderpath = targetfolderpath + firstfolder
    secondfolder = '%06d/' % shotno
    targetfolderpath = targetfolderpath + secondfolder
    subfolder = '%06d' % subshot
    targetfolderpath = targetfolderpath + subfolder
    #print targetfolderpath
    filelist = ftpList(targetfolderpath)
    size = ftpSize(targetfolderpath)
    if size < 1000:
    #if 0 == len(filelist):
        return False
    return True

if __name__ == "__main__":
    import time
    start = time.time()
    md4GAN = MakeDataset4GAN(ShotNo=146021, mode="ECHsetting_estimation")
    #md4GAN.ana_plot_eg(arr_time=[3.083])
    #md4GAN.ana_plot_LHDGauss_deprof(arr_time=[3.083, 4.0])
    #md4GAN.searchData()
    md4GAN.get_dataseries()
    #md4GAN.ana_plot_ece(arr_time=[4.3])
    #md4GAN.ana_plot_timetrace(arr_time=[4.3])
    val_time=3.8
    #md4GAN.ana_plot_LHDGauss_deprof_3D(val_time=val_time)
    #md4GAN.ana_plot_LHDGauss_deprof(arr_time=[3.800, 4.0])
    #md4GAN.plot_tsmap_LHDGAUSS_input(val_time=val_time)
    #md4GAN.ana_plot_timetrace(arr_time=[3.8])
    #md4GAN.ana_plot_timetrace_TS()
    #md4GAN.ana_plot_eg(arr_time=[3.8])
    #md4GAN.make_dataset(153377)
    #md4GAN.searchData()
    #i = 150003
    #for i in range(153367, 161167):
    #for i in range(153407, 161167):
    #md4GAN.ana_plot_tsmap_3D(val_time)
    t = time.time() - start
    print("erapsed time= %.3f sec" % t)
