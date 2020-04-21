#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from nifs.voltdata import *
import numpy
import matplotlib.pyplot as plt

data=VoltData.retrieve('Bolometer',48000,1,1)
voltdata = data.get_val()
clk = data.params['ClockSpeed']
timedata = numpy.arange(0.0, len(voltdata)) / clk

plt.title('Bolometer')
plt.xlabel('t [sec]')
plt.ylabel('v [V]')
plt.plot(timedata,voltdata)
plt.show()
