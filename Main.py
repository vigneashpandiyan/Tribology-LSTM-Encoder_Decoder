# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:14:49 2022

@author: srpv

- Main block for replicating the methodology proposed in the following work.

Pandiyan, Vigneashwara, Mehdi Akeddar, Josef Prost, Georg Vorlaufer, Markus Varga, and Kilian Wasmer. 
"Long short-term memory based semi-supervised encoder—decoder for early prediction of failures in self-lubricating bearings."
 Friction (2022): 1-16.
 
 
- Required [groud-truths, corresponding windows]
"""
#%%

import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
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
from Network import *
from Visualization_Utils import *

rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting Up CUDA
Epochs=2

#%% Data Preparation

'''
The script is hard coded based on ground-truths...
- The ground truths [windows] are running-in, normal and anomaly
'''

Force = np.load('Force.npy') #Data source
#Setting up a dataframe

Running_in=Force[0:4000,:]
Running_in = pd.DataFrame(Running_in)
Running_in.shape

normal_df = Force[8000:18000,:]
normal_df = pd.DataFrame(normal_df)
normal_df.shape

anomaly_df=Force[23000:34000,:]
anomaly_df = pd.DataFrame(anomaly_df)
anomaly_df.shape


#%%Data splitting

'''
The idea is to train the model on the normal data space.
So we will work on the windows in the normal regime.
'''

train_df, val_df = train_test_split(
  normal_df,
  test_size=0.33,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33, 
  random_state=RANDOM_SEED
)


#%% Data loader 

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)

normal_dataset, _, _ = create_dataset(normal_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)
Running_in_dataset, _, _ = create_dataset(Running_in)

#%% Model initialization

class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=128):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x1 = self.encoder(x)
    x = self.decoder(x1)
    return x,x1

#%% Model training

model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)

model, history = train_model(
  model, 
  train_dataset, 
  val_dataset, 
  n_epochs=Epochs #Epochs
)

#%% Saving the model

MODEL_PATH = 'model.pth'
torch.save(model, MODEL_PATH)

#%%
plot_learning_curves(history)
#%% Compute the threshold

_,_, train_dataset_losses = predict(model, train_dataset)
THRESHOLD=Compute_threshold(model, train_dataset)
print(THRESHOLD)
plot_loss_distribution(train_dataset_losses,THRESHOLD,"green","Train_dataset_losses")

#%% Plots 

predictions,_, test_normal_dataset_losses = predict(model, test_normal_dataset)
correct = sum(l <= THRESHOLD for l in test_normal_dataset_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')
plot_loss_distribution(test_normal_dataset_losses,THRESHOLD,"orange","Test_normal_dataset_losses")

predictions,_, normal_dataset_losses = predict(model, normal_dataset)
correct = sum(l <= THRESHOLD for l in normal_dataset_losses)
print(f'Correct normal predictions: {correct}/{len(normal_dataset)}')
plot_loss_distribution(normal_dataset_losses,THRESHOLD,"purple","Normal_dataset_losses")

anomaly_dataset = test_anomaly_dataset
predictions,_,anomaly_dataset_losses = predict(model, anomaly_dataset)
correct = sum(l > THRESHOLD for l in anomaly_dataset_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')
plot_loss_distribution(anomaly_dataset_losses,THRESHOLD,"blue","Anomaly_dataset_losses")

Running_in_dataset = Running_in_dataset
predictions,_, Running_in_dataset_losses = predict(model, Running_in_dataset)
correct = sum(l > THRESHOLD for l in Running_in_dataset_losses)
print(f'Correct anomaly predictions: {correct}/{len(Running_in_dataset)}')
plot_loss_distribution(Running_in_dataset_losses,THRESHOLD,"yellow","Running_in_dataset_losses")


#%% Results on Recontruction

plot_prediction_results(Running_in_dataset,test_normal_dataset,test_anomaly_dataset, model)

#%% Latent Space Tsne Visualization

'''
Windows for respective windows are hardcoded based on ground-truths
'''

Running_in=tsne_dataset('Force.npy',1000,4000,0)  # Dataset, the start window no, Ending window no, label
Stable=tsne_dataset('Force.npy',11000,14000,1)  # Dataset, the start window no, Ending window no, label
Anomaly=tsne_dataset('Force.npy',28000,31000,2) # Dataset, the start window no, Ending window no, label


Dataset_Tsne = [Running_in, Stable, Anomaly]
Dataset_Tsne = pd.concat(Dataset_Tsne)
Tsne_labels = Dataset_Tsne.iloc[: , -1]
Dataset_Tsne = Dataset_Tsne.iloc[:, :-1]


Dataset_Tsne, seq_len, n_features = create_dataset(Dataset_Tsne)
_,latent,_ = predict(model, Dataset_Tsne)
Dataset_Tsne = np.array(latent)


perp=40
graph_name= '3D_tsne'+'.png'
ax,fig,tsne_fit,target=TSNEplot(Dataset_Tsne,Tsne_labels,graph_name,str('3D_tsne'),perp)
graph_name= 'tsne'+'.gif'

def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))


graph_name_2D='Tsne_Feature_2D' +'_'+str(perp)+'.png'
graph_title = "Feature space distribution"
plot_embeddings(tsne_fit, target,graph_name_2D,graph_title)
np.save('Embedding_space.npy',Dataset_Tsne)
np.save('Embedding_labels.npy',Tsne_labels)


#%%

PATH = 'model.pth'

model_train = torch.load(PATH)
model.eval()

#%%

reconstruction_loss(model, str('Force.npy'))
reconstruction_loss_scatter(model, str('Force.npy'),THRESHOLD,8000,18000)


