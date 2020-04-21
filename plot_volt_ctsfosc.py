#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from nifs.voltdata import *
import numpy
import matplotlib.pyplot as plt

data=VoltData.retrieve('CTSfosc1',131030,1,1)
voltdata = data.get_val()
#clk = data.params['ClockSpeed']
timedata = numpy.arange(0.0, len(voltdata))# / clk

plt.title('CTSfosc1')
plt.xlabel('t [sec]')
plt.ylabel('v [V]')
idx = 10000
plt.plot(timedata[:idx],voltdata[:idx])
plt.show()
