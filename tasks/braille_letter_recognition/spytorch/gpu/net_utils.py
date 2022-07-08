import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import random
import json

from sklearn.metrics import confusion_matrix


device = torch.device("cuda:0")

dtype = torch.float

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

threshold = 1 

file_dir_data = './' #'./reading/'
file_type = 'data'
file_thr = str(threshold)
file_ref = 'Null'
file_name = file_dir_data + file_type + '_th' + file_thr + '_rp' + file_ref

file_dir_params = './' #'./net_params/'
param_filename = 'parameters_th1'
file_name_parameters = file_dir_params + param_filename + '.txt'
params = {}
with open(file_name_parameters) as file:
  for line in file:
    (key, value) = line.split()
    if key == 'time_bin_size' or key == 'nb_input_copies':
      params[key] = int(value)
    else:
      params[key] = np.double(value)

file_dir_layers = './' #'./trained/'
layers_filename = 'layers_th1'
file_name_layers = file_dir_layers + layers_filename + '.pt'

class SurrGradSpike(torch.autograd.Function):
  """Zenke & Ganguli (2018)"""

  scale = params['scale']

  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    out = torch.zeros_like(input)
    out[input > 0] = 1.0
    return out

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
    return grad

spike_fn  = SurrGradSpike.apply


def demo(params, file_name, letter_written=letters, taxels=None):
    
    max_time = int(54*25) #ms
    time_bin_size = int(params['time_bin_size']) # ms
    global time
    time = range(0,max_time,time_bin_size)
    ## Increase max_time to make sure no timestep is cut due to fractional amount of steps
    global time_step
    time_step = time_bin_size*0.001
    global data_steps
    data_steps = len(time)
    
    infile = open(file_name, 'rb')
    data_dict = pickle.load(infile)
    infile.close()
    # Extract data
    data = []
    labels = []
    bins = 1000  # [ms] 
    nchan = len(data_dict[1]['events']) # number of channels/sensors
    global nb_channels
    nb_channels = nchan
    nb_repetitions = 200
    idx = random.randrange(0,len(data_dict)-1)
    dat = data_dict[idx]['events'][:]
    events_array = np.zeros([nchan,round((max_time/time_bin_size)+0.5),2])
    for taxel in range(len(dat)):
        for event_type in range(len(dat[taxel])):
            if dat[taxel][event_type]:
                indx = bins*(np.array(dat[taxel][event_type]))
                indx = np.array((indx/time_bin_size).round(), dtype=int)
                events_array[taxel,indx,event_type] = 1
    if taxels != None:
        events_array = np.reshape(np.transpose(events_array, (1,0,2))[:,taxels,:],(events_array.shape[1],-1))
        selected_chans = 2*len(taxels)
    else:
        events_array = np.reshape(np.transpose(events_array, (1,0,2)),(events_array.shape[1],-1))
        selected_chans = 2*nchan
    reading = events_array
    label = letter_written[letter_written.index(data_dict[idx]['letter'])]
        
    return torch.tensor(reading[None, :, :],dtype=dtype), label


def run_snn(inputs, layers, device=device):

    bs = inputs.shape[0]
    h1_from_input = torch.einsum("abc,cd->abd", (inputs.tile((nb_input_copies,)), layers[0]))
    syn = torch.zeros((bs,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((bs,nb_hidden), device=device, dtype=dtype)

    out = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)

    # Here we define two lists which we use to record the membrane potentials and output spikes
    mem_rec = []
    spk_rec = []

    # Compute hidden (recurrent) layer activity
    for t in range(nb_steps):
        h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, layers[2]))
        mthr = mem-1.0
        out = spike_fn(mthr)
        rst = out.detach() # We do not want to backprop through the reset

        new_syn = alpha*syn + h1
        new_mem = (beta*mem + syn)*(1.0-rst)

        mem_rec.append(mem)
        spk_rec.append(out)
    
        mem = new_mem
        syn = new_syn

    # Now we merge the recorded membrane potentials into a single tensor
    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, layers[1]))
    flt = torch.zeros((bs,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((bs,nb_outputs), device=device, dtype=dtype)
    s_out_rec = [out] # out is initialized as zeros, so it is fine to start with this
    out_rec = [out]
    for t in range(nb_steps):
        mthr_out = out-1.0
        s_out = spike_fn(mthr_out)
        rst_out = s_out.detach()

        new_flt = alpha*flt + h2[:,t]
        new_out = (beta*out + flt)*(1.0-rst_out)

        flt = new_flt
        out = new_out

        out_rec.append(out)
        s_out_rec.append(s_out)

    out_rec = torch.stack(out_rec,dim=1)
    s_out_rec = torch.stack(s_out_rec,dim=1)
    other_recs = [mem_rec, spk_rec, s_out_rec]
    layers_update = layers

    
    return out_rec, other_recs, layers_update


def load_layers(file, map_location=device, requires_grad=True, variable=False):
    
    if variable:
        
        lays = file
        
        for ii in lays:
            ii.requires_grad = requires_grad
    
    else:
        
        lays = torch.load(file, map_location=map_location)
    
        for ii in lays:
            ii.requires_grad = requires_grad
        
    return lays


def build_and_predict(params, x, device=device):
    
    x = x.to(device)
    
    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel
    global nb_inputs
    nb_inputs  = nb_channels*nb_input_copies
    global nb_hidden
    nb_hidden  = 450
    global nb_outputs
    nb_outputs = len(np.unique(letters))+1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem'] # ms
    tau_syn = tau_mem/params['tau_ratio']
    
    global alpha
    alpha   = float(np.exp(-time_step/tau_syn))
    global beta
    beta    = float(np.exp(-time_step/tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor']*fwd_weight_scale

    # Spiking network
    layers = load_layers(file_name_layers, map_location=device)
    
    # Make predictions
    output, others, _ = run_snn(x,layers,device=device)
    
    ### Classification through spikes
    m = torch.sum(others[-1],1) # sum over time
    _, am = torch.max(m,1) # argmax over output units
    #################################
    
    return letters[am.detach().cpu().numpy()[0]], output, others


