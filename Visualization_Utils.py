# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:09:16 2021

@author: srpv
"""


import torch

import copy
import numpy as np
import pandas as pd
import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim

import torch.nn.functional as F
import matplotlib.patches as mpatches
from matplotlib import cm
from Utils import *
from tSNE import *

# rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail 


def reconstruction_loss(model, force_df):
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    tail=path_leaf(force_df)
    A=tail.split('.')
    A=A[0]
    
    print(A)
    
    filename='Reconstruction_loss_distribution_'+str(A)+'.png'
    filename2='Reconstruction_loss_Histogram_distribution'+str(A)+'.png'
    
    force_df  = np.load(force_df)
    force_df = pd.DataFrame(force_df)
    
    print(force_df.shape)
    
    force_dataset, seq_len, n_features = create_dataset(force_df)
    _,_,force_losses = predict(model, force_dataset)
    losses = np.array(force_losses)
    
    
    windowsize= len(losses) #window size you wish to visualise
    sample_rate= 1
    dt=1/sample_rate
    t0=0
    time = np.arange(0, windowsize) * dt + t0
    time = np.expand_dims(time, axis=1)
    
    losses=losses.squeeze()
    time=time.squeeze()
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    plt.vlines(time,ymin=0, ymax=losses, color='blue', alpha=1,label="Reconstruction error")
    plt.ylabel('Reconstruction error',fontsize=15)
    plt.xlabel('cycle no',fontsize=15)
    plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
    plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plt.savefig(filename,dpi=800)
    plt.show()
    plt.clf()
    

    
    
#%%

def reconstruction_loss_scatter(model, force_df,THRESHOLD,n1,n2):
    
    tail=path_leaf(force_df)
    A=tail.split('.')
    A=A[0]
    
    print(A)
    
    filename1='Error_scatter_distribution_'+str(A)+'.png'
    filename2='Error_scatter_bar_'+str(A)+'.png'
    
    force_df  = np.load(force_df)
    force_df = pd.DataFrame(force_df)
    
    print(force_df.shape)
    
    force_dataset, seq_len, n_features = create_dataset(force_df)
    _,_,force_losses = predict(model, force_dataset)

    losses = np.array(force_losses)
    anomaly_indices = []
    anomaly_values = []
    error_values = []
    for i in range(losses.shape[0]):
        error=losses[i]-THRESHOLD
        error_values.append(error)
        if i > n2 or i < n1 :
            # print(i)
            if losses[i] >= THRESHOLD:
                anomaly_indices.append(i)
                anomaly_values.append(losses[i])
                
            
    plt.rcParams.update(plt.rcParamsDefault)   
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(losses, label="Reconstruction error",color='blue',linewidth=0.75)
    plt.scatter(x=anomaly_indices, y=anomaly_values, color='red', label="Anomaly")
    plt.axhline(y=THRESHOLD, c='r', linestyle='--',linewidth=0.5)
    plt.legend(loc = "upper left")
    plt.savefig(filename1,dpi=800)
    plt.show()
    plt.clf()
    
    
    losses = np.array(force_losses)
    losses = pd.DataFrame(losses)
    losses=losses.set_axis(['Reconstruction_error'], axis=1)
    
    
    plt.rcParams.update(plt.rcParamsDefault)   
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax = losses.plot(kind='bar',color='blue')
    plt.scatter(x=anomaly_indices, y=anomaly_values, color='red', label="Anomaly")
    plt.axhline(y=THRESHOLD, c='r', linestyle='--',linewidth=0.5)
    plt.legend(loc = "upper left")
    ax.locator_params(nbins=10, axis='x')
    plt.ylabel('Reconstruction error',fontsize=15)
    plt.xlabel('Cycle no',fontsize=15)
    plt.title('Reconstruction error over cycle',fontsize=20)
    plt.savefig(filename2,dpi=800)
    plt.show()
    plt.clf()

#%%


def plotsignal(li,color,filename,add):
    
    plt.rcParams.update(plt.rcParamsDefault)
    windowsize= len(li) #window size you wish to visualise
    sample_rate= 100
    dt=1/sample_rate
    t0=0
    time = np.arange(0, windowsize) * dt + t0
    time=time+add
    fig, ax = plt.subplots(figsize=(10, 3))
    plt.plot(time,li,color=color, linewidth=3)
    plt.ylabel('Amplitude (V)',fontsize=15)
    plt.xlabel('cycle no',fontsize=15)
    plotname='Force_cycle_'+str(filename)+'.png'
    plt.savefig(plotname,bbox_inches='tight',pad_inches=0.1,dpi=200)
    plt.show()
    plt.clf()



def plotsignals(Running_in,Stable,Anomaly):
    
    plt.rcParams.update(plt.rcParamsDefault)
    windowsize= len(Running_in)
    sample_rate= 100
    dt=1/sample_rate
    t0=0
    time = np.arange(0, windowsize) * dt + t0
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    plt.plot(time,Running_in,color="#FF1818", linewidth=2,label='Running_in')
    plt.plot(time,Stable,color="#00CB26", linewidth=2,label='Stable')
    plt.plot(time,Anomaly,color="#0000EC", linewidth=2,label='Anomaly')
    plt.ylabel('Amplitude (V)',fontsize=15)
    plt.xlabel('cycle no',fontsize=15)

    plotname='Force_cycle_comparison'+'.png'
    plt.legend(loc='upper right',bbox_to_anchor=(1.20, 1))
    plt.savefig(plotname,bbox_inches='tight',pad_inches=0.1,dpi=200)
    plt.show()


