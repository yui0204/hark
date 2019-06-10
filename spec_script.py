# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:12:45 2018

@author: yui_sudo

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spec = pd.read_csv("spec.txt", delimiter=" ", header=None) # 日本語パス不可
spec = spec.T.iloc[2:-1]

spec = np.array(spec, np.float32)

#plt.pcolormesh(spec)
plt.pcolormesh(np.arange(0,4.1,0.1), np.arange(0,359,5), spec)
#plt.pcolormesh(np.arange(0,4.1,0.1), np.arange(0,359,45), spec)
plt.colorbar()