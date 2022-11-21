# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:18 2021

@author: srpv
"""

import numpy as np
import torch
import copy
from torch import nn, optim
import torch.nn.functional as F
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import pandas as pd
import ntpath

#%%

'''
Dataset preparation--> to torch
'''

def create_dataset(df):
    


  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features

#%%

'''
Model training

-- Optimizer
-- Learning rate scheduler
-- Cost/ Loss function
'''

def train_model(model, train_dataset, val_dataset, n_epochs):
    
    
  
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  
  scheduler = StepLR(optimizer, step_size = 20, gamma= 0.25 )
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    scheduler.step()
    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred,_ = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred,_ = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history

#%%

'''
Plot learning curves
'''

def plot_learning_curves(history):
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots()
    ax.plot(history['train'],color="b", linewidth=2.5,label='Training loss')
    ax.plot(history['val'],color="g", linewidth=2.5,label='Validation loss')
    plt.rc('legend')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'],loc='upper right', borderpad=1)
    plt.title('Loss over training epochs')
    plt.savefig('Training_Loss.png',dpi=800,bbox_inches='tight')
    plt.show();
#%%

'''
Compute the limits
'''

def Compute_threshold(model, train_dataset):
    
    _,_, losses = predict(model, train_dataset)
    scores_normal = np.asarray(losses)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    THRESHOLD = normal_avg + (normal_std * 3)

    return THRESHOLD

#%%

'''
Plot the model prediction against the limits
'''

def plot_loss_distribution(losses,THRESHOLD,color,graph_name):
    
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots()
    sns.distplot(losses, bins=50,rug_kws={"color": color}, kde=True,color=color);
    plt.axvline(x=THRESHOLD, c='r', linestyle='--',linewidth=4)
    plt.xlabel('Reconstruction loss')
    plt.title(f'Threshold :{THRESHOLD}')
    graph=graph_name+'.png'
    plt.savefig(graph,dpi=400,bbox_inches='tight')
    plt.show()
    plt.clf()


#%%

'''
Helper function for plot_prediction_results
'''
def plot_prediction(data, model, title, ax,color):
  predictions,_, pred_losses = predict(model, [data])

  ax.plot(data, 'black' , label='Original',linewidth=2)
  ax.plot(predictions[0],linestyle='--',linewidth=4,color=color, label='LSTM Reconstruction')
  ax.tick_params(axis='both', which='major', labelsize=30)
  ax.tick_params(axis='both', which='minor', labelsize=30)
  ax.ticklabel_format(useOffset=False)
  ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})',fontsize=25)
  plt.rc('legend',**{'fontsize':12})
  # ax.legend(loc='upper right')



#%%

def plot_prediction_results(Running_in_dataset,test_normal_dataset,test_anomaly_dataset, model):
    fig, axs = plt.subplots(
      nrows=3,
      ncols=4,
      sharey=True,
      sharex=True,
      figsize=(22, 12)
    )
    
    
    for i, data in enumerate(Running_in_dataset[:4]):
      plot_prediction(data, model, title='Running_in', ax=axs[0, i],color="#FF1818")
    # axs[2].legend()
    for i, data in enumerate(test_normal_dataset[:4]):
      plot_prediction(data, model, title='Normal', ax=axs[1, i],color="#00CB26")
    # axs[0].legend()
    for i, data in enumerate(test_anomaly_dataset[:4]):
      plot_prediction(data, model, title='Anomaly', ax=axs[2, i],color="#0000EC")
    # axs[1].legend() 
    fig.tight_layout();
    plt.savefig('Normal and Anomaly.png',dpi=800)
    plt.show()
    plt.clf()



#%%

'''
function for computing the reconstruction, latent space and reconstruction loss
'''

def predict(model, dataset):
  predictions, losses,latent = [], [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred,latent_space = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      latent.append(latent_space.cpu().numpy().flatten())
      
      losses.append(loss.item())
  return predictions,latent, losses

