# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:18 2021

@author: srpv

The following script helps to compute t-sne embeddings based on the ....
latent space vectors from the bottle neck layer of the Encoder- Decoder architecture
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(1974)

#%%

'''
Data set preparation
- Hard coded limits
- Hard coded labels
'''

def tsne_dataset(dataset,start,end,label):

    data = np.load(dataset)
    data = data[start:end,:] 
    columns = len(data)
    z=np.full(shape=columns, fill_value=label)
    z=np.expand_dims(z, axis=1) 
    data=np.append(data, z, axis=1)
    data = pd.DataFrame(data)
    
    return data


#%%

'''
Computing t-sne and ploting in 3D
'''

def TSNEplot(output,target,graph_name,graph_title,perplexity):
    
    
    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ',perplexity)
    
    target=target
    target = np.ravel(target)
    
    RS=np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)
    np.save('tsne_3d.npy',tsne_fit)
    tsne_fit = np.load('tsne_3d.npy')
    np.save('target.npy',target)
    target = np.load('target.npy')
    
    '''
    t-sne computed
    
    '''
    
    df2 = pd.DataFrame(target) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'Running_in')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'Normal')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Anomaly')
    target = pd.DataFrame(df2)
    target = target.to_numpy()
    target = np.ravel(target)
    
   
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    x3=tsne_fit[:, 2]
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=target))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    uniq=["Running_in","Normal","Anomaly"]
    z = range(1,len(uniq))
    
    '''
    Plotting 3D 
    '''
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(12,9), dpi=300)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(azim=115)#115
    marker= ["*",">","X","o","s"]
    color = [ 'r','g','b','orange','purple']
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
   
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    

    ax.set_ylim(min(x2), max(x2))
    ax.set_zlim(min(x3), max(x3))
    ax.set_xlim(min(x1), max(x1))
    
    for i in range(len(uniq)):
        
        indx = (df['label']) == uniq[i]
        
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        ax.plot(a, b, c ,color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=5)
      

    plt.xlabel ('Dimension 1', labelpad=25)
    plt.ylabel ('Dimension 2', labelpad=25)
    ax.set_zlabel('Dimension 3',labelpad=25)
    plt.title(graph_title,fontsize = 30)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
   
    ax.tick_params(axis='z', labelsize=16)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name, bbox_inches='tight',dpi=200)
    plt.show()
    
    return ax,fig,tsne_fit,target

def plot_embeddings(tsne_fit, targets,graph_name_2D,graph_title, xlim=None, ylim=None):
    plt.rcParams.update(plt.rcParamsDefault)
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    
    group=targets
    
    df2 = pd.DataFrame(group) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'Running_in')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'Normal')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Anomaly')
    group = pd.DataFrame(df2)
    group = group.to_numpy()
    group = np.ravel(group)
    
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')
    
    
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    uniq=["Running_in","Normal","Anomaly"]
    
    z = range(1,len(uniq))
       
    
    fig = plt.figure(figsize=(12,9), dpi=100)
    
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)
    
    marker= ["*",">","X","o","s"]
    color = [ 'r','g','b','orange','purple']
    
    for i in range(len(uniq)):
        indx = (df['label']) == uniq[i]
        a=x1[indx]
        b=x2[indx]
        plt.plot(a, b, color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=8)
        
        
    
    plt.xlabel ('Dimension 1')
    plt.ylabel ('Dimension 2')
    plt.title(graph_title,fontsize = 30)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)

    plt.legend(uniq,bbox_to_anchor=(1.35, 1.05))
    plt.savefig(graph_name_2D, bbox_inches='tight',dpi=100)
    
    plt.show()


