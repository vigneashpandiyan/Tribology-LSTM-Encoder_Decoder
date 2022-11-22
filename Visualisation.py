# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:22:39 2021

@author: srpv
"""

import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
from matplotlib import cm
from Visualization_Utils import *

COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Force = np.load('Force.npy')
Total_cycle_force =Force.flatten()


plotsignal(Total_cycle_force,filename="whole",color="#000000",add=0) 

#%%

Running_in = Force[1000:1010,:]
Running_in=Running_in.flatten()

Stable = Force[9000:9010,:]
Stable=Stable.flatten()

Anomaly = Force[30000:30010,:]
Anomaly=Anomaly.flatten()

plotsignal(Running_in,filename="Running_in",color="#FF1818",add=1000) 
plotsignal(Stable,filename="Stable",color="#00CB26",add=9000)
plotsignal(Anomaly,filename="Anomaly",color="#0000EC",add=30000) 

#%%

plotsignals(Running_in,Stable,Anomaly)

#%%



