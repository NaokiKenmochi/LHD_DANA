#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nifs.rawdata import *
import matplotlib.pyplot as plt #グラフ描画ライブラリの読み込み

# Bolomgeter の ショット番号=48000, サブショット番号=1, チャンネル番号=1 のデータを取得
data=RawData.retrieve('Bolometer',48000,1,1)

range = data.params['Range'] # レンジ取得
rangefactor = data.params['RangeFactor'] # 倍率
resolution = data.params['Resolution(bit)'] # 解像度(bit数)
volt = (range * rangefactor * 2 ) / ( 2 ** resolution ) # 生データを電圧に変換する際の係数

val = data.get_val() # 生データの取得 (デジタイザーのイメージデータを int の配列にしたもの)

# オフセットを引いて先にもとめた係数をかけ、電圧の配列を得る
voltdata = (val - ( (2 ** resolution) / 2 )) * volt 
clk = data.params['ClockSpeed'] # サンプリングクロックの取得

# 電圧データと同じ長さの配列を作り、サンプリングクロックで割る(サンプリング時刻の配列を得る)
timedata = numpy.arange(0.0, len(val)) / clk 

plt.title('Bolometer') # グラフタイトルセット
plt.xlabel('t [sec]') # X-軸ラベルのセット
plt.ylabel('v [V]') # y-軸ラベルのセット
plt.plot(timedata,voltdata) # 描画
plt.show()
