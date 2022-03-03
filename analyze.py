import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
from itertools import repeat
from multiprocessing import Pool

import matplotlib.pyplot as plt

PATH = "/home/rudedaisy/detectron2/export/"
BINS = 100
SAMPLES = 500
MODE = 5
MODE_DEFS = {0: 'single', 
             1: 'layer-wise',
             2: 'single_image',
             3: 'total',
             4: 'weights_only',
             5: 'correlate_ifm_offset_finegrain'}
ifm_log = "ifm_log"
f_model = "model.csv"

def count_layers():
    # Count the number of offset maps in the model
    num_layers = 0
    f = open(os.path.join(PATH, f_model), "r")
    for line in f:
        if "conv2_offset" in line:
            num_layers += 1
    f.close
    return num_layers

def get_strides_pads_kdims():
    strides = []
    pads = []
    kdims = []
    f = open(os.path.join(PATH, f_model), "r")
    for line in f:
        if "conv2_offset" in line:
            stride = int(line.split(',')[2])
            pad = int(line.split(',')[3])
            kdim = int(line.split(',')[6])
            strides.append(stride)
            pads.append(pad)
            kdims.append(kdim)
    f.close
    return strides, pads, kdims

def stat(fileName, data):
    ofm = np.load(fileName)
    #print(ofm.shape)
    ofm = list(ofm.flatten())
    data += ofm
    return data

def run_weights_only(num_layers):
    data = []
    for i in range(num_layers):
        data = stat(os.path.join(PATH, "weight-" + str(i) + ".npy"), data)
    mean = np.mean(data)
    std = np.std(data)
    print("Weight mean", mean)
    print("Weight std", std)

def count_ifms_imgs():
    # Count the number of images processed
    num_ifms = 0
    f = open(os.path.join(PATH, ifm_log), "r")
    for line in f:
        if len(line) > 2:
            num_ifms += 1
    f.close()
    #assert num_ifms % num_layers == 0
    num_imgs = num_ifms // num_layers

    return (num_ifms, num_imgs)

def summary(imgs=[], layers=[]):
    data = []
    for i in imgs:
        for l in layers:
            data = stat(os.path.join(PATH, "OFM-"+str(i)+"-"+str(l)+".npy"), data)
    
    plt.hist(data, density=True, bins = BINS)
    plt.ylabel('Probability')
    plt.xlabel('Offset')
    plt.show()

    mean = np.mean(data)
    std = np.std(data)
    print("Mean", mean)
    print("STD", std)

def populate_finegrain_stat(ifm_data_unfold, wgt_data, ofm_data, ifm_stats, ifm_wgt_stats, ofm_stats):
    # unfold shape: batch_size, channels, h_windows, w_windows, kh, kw
    # wgt shape: offsets=kh*kw*2, channels, kh, kw
    # ofm shape: batch_size, offsets=kh*kw*2, h_windows, w_windows
    num_datapoints = ifm_data_unfold.shape[0] * ifm_data_unfold.shape[2] * ifm_data_unfold.shape[3]

    ifm_sum = ifm_data_unfold.sum(axis=(1, 4, 5)).flatten().tolist()
    ifm_mean = ifm_data_unfold.mean(axis=(1, 4, 5)).flatten().tolist()
    ifm_std = ifm_data_unfold.std(axis=(1, 4, 5)).flatten().tolist()
    ifm_absmax = np.amax(np.abs(ifm_data_unfold), axis=(1, 4, 5)).flatten().tolist()
    num_reduced_elems = ifm_data_unfold.shape[1] * ifm_data_unfold.shape[4] * ifm_data_unfold.shape[5]
    ifm_density = (np.count_nonzero(ifm_data_unfold, axis=(1, 4, 5)) / num_reduced_elems).flatten().tolist()
    ifm_stats['sum'] += ifm_sum
    ifm_stats['mean'] += ifm_mean
    ifm_stats['std'] += ifm_std
    ifm_stats['absmax'] += ifm_absmax
    ifm_stats['density'] += ifm_density
    ifm_stats['channels'] += [ifm_data_unfold.shape[1]] * num_datapoints
    ifm_stats['shape'] += [ifm_data_unfold.shape[2] * ifm_data_unfold.shape[3]] * num_datapoints

    wgt_sum = [wgt_data.sum()] * num_datapoints
    wgt_mean = [wgt_data.mean()] * num_datapoints
    wgt_std = [wgt_data.std()] * num_datapoints
    wgt_absmax = [np.amax(np.abs(wgt_data))] * num_datapoints
    num_reduced_elems = np.prod(wgt_data.shape)
    wgt_density = [np.count_nonzero(wgt_data) / num_reduced_elems] * num_datapoints
    ifm_wgt_stats['sum_sum'] += (np.multiply(ifm_sum, wgt_sum)).tolist()
    ifm_wgt_stats['sum_mean'] += (np.multiply(ifm_sum, wgt_mean)).tolist()
    ifm_wgt_stats['sum_std'] += (np.multiply(ifm_sum, wgt_std)).tolist()
    ifm_wgt_stats['sum_absmax'] += (np.multiply(ifm_sum, wgt_absmax)).tolist()
    ifm_wgt_stats['sum_density'] += (np.multiply(ifm_sum, wgt_density)).tolist()
    ifm_wgt_stats['mean_sum'] += (np.multiply(ifm_mean, wgt_sum)).tolist()
    ifm_wgt_stats['mean_mean'] += (np.multiply(ifm_mean, wgt_mean)).tolist()
    ifm_wgt_stats['mean_std'] += (np.multiply(ifm_mean, wgt_std)).tolist()
    ifm_wgt_stats['mean_absmax'] += (np.multiply(ifm_mean, wgt_absmax)).tolist()
    ifm_wgt_stats['mean_density'] += (np.multiply(ifm_mean, wgt_density)).tolist()
    ifm_wgt_stats['std_sum'] +=	(np.multiply(ifm_std, wgt_sum)).tolist()
    ifm_wgt_stats['std_mean'] += (np.multiply(ifm_std, wgt_mean)).tolist()
    ifm_wgt_stats['std_std'] +=	(np.multiply(ifm_std, wgt_std)).tolist()
    ifm_wgt_stats['std_absmax'] += (np.multiply(ifm_std, wgt_absmax)).tolist()
    ifm_wgt_stats['std_density'] += (np.multiply(ifm_std, wgt_density)).tolist()
    ifm_wgt_stats['absmax_sum'] += (np.multiply(ifm_absmax, wgt_sum)).tolist()
    ifm_wgt_stats['absmax_mean'] += (np.multiply(ifm_absmax, wgt_mean)).tolist()
    ifm_wgt_stats['absmax_std'] += (np.multiply(ifm_absmax, wgt_std)).tolist()
    ifm_wgt_stats['absmax_absmax'] += (np.multiply(ifm_absmax, wgt_absmax)).tolist()
    ifm_wgt_stats['absmax_density'] += (np.multiply(ifm_absmax, wgt_density)).tolist()
    ifm_wgt_stats['density_sum'] += (np.multiply(ifm_density, wgt_sum)).tolist()
    ifm_wgt_stats['density_mean'] += (np.multiply(ifm_density, wgt_mean)).tolist()
    ifm_wgt_stats['density_std'] += (np.multiply(ifm_density, wgt_std)).tolist()
    ifm_wgt_stats['density_absmax'] += (np.multiply(ifm_density, wgt_absmax)).tolist()
    ifm_wgt_stats['density_density'] += (np.multiply(ifm_density, wgt_density)).tolist()
    
    ofm_stats['sum'] += ofm_data.sum(axis=1).flatten().tolist()
    ofm_stats['mean'] += ofm_data.mean(axis=1).flatten().tolist()
    ofm_stats['std'] += ofm_data.std(axis=1).flatten().tolist()
    ofm_stats['absmax'] += np.amax(np.abs(ofm_data), axis=1).flatten().tolist()
    #num_reduced_elems = ofm_data.shape[1]
    #ofm_stats['density'] += (np.count_nonzero(ofm_data, axis=1) / num_reduced_elems).flatten().tolist()

    return ifm_stats, ifm_wgt_stats, ofm_stats

# Parallelize with threads                                                                                                                                                                                                                                                     
def _local_extract(imgs=[], layers=[], strides=[], pads=[], kdims=[]):
    ifm_stats = {'sum':[],
                 'mean':[],
                 'std':[],
                 'absmax':[],
                 'density':[],
                 'channels':[],
                 'shape':[]}
    ifm_wgt_stats = {'sum_sum':[],
                     'sum_mean':[],
                     'sum_std':[],
                     'sum_absmax':[],
                     'sum_density':[],
                     'mean_sum':[],
                     'mean_mean':[],
                     'mean_std':[],
                     'mean_absmax':[],
                     'mean_density':[],
                     'std_sum':[],
                     'std_mean':[],
                     'std_std':[],
                     'std_absmax':[],
                     'std_density':[],
                     'absmax_sum':[],
                     'absmax_mean':[],
                     'absmax_std':[],
                     'absmax_absmax':[],
                     'absmax_density':[],
                     'density_sum':[],
                     'density_mean':[],
                     'density_std':[],
                     'density_absmax':[],
                         'density_density':[]}
    ofm_stats = {'sum':[],
                 'mean':[],
                 'std':[],
                 'absmax':[],
                 'density':[]}
    for i in imgs:
        for l in layers:
            stride = strides[l]
            pad = pads[l]
            kdim = kdims[l]
            ifm_data = np.load(os.path.join(PATH, "IFM-"+str(i)+"-"+str(l)+".npy"))
            wgt_data = np.load(os.path.join(PATH, "weight-" + str(l) + ".npy"))
            ofm_data = np.load(os.path.join(PATH, "OFM-"+str(i)+"-"+str(l)+".npy"))
            #print(ifm_data.shape)
            #print(ofm_data.shape)
            
            # unfold shape: batch_size, channels, h_windows, w_windows, kh, kw
            ifm_data_unfold = np.array(F.pad(torch.tensor(ifm_data),(pad, pad, pad, pad, 0, 0, 0, 0)).unfold(2, kdim, stride).unfold(3, kdim, stride))
            #print(ifm_data_unfold.shape)
            
            ifm_stats, ifm_wgt_stats, ofm_stats = populate_finegrain_stat(ifm_data_unfold, wgt_data, ofm_data, ifm_stats, ifm_wgt_stats, ofm_stats)
    return ifm_stats, ifm_wgt_stats, ofm_stats

def correlate_finegrain(imgs=[], layers=[], strides=[], pads=[], kdims=[]):

    with Pool(22) as p:
        imgs_unroll = [x for item in imgs for x in repeat(item, len(layers))]
        layers_unroll = layers*len(imgs)
        strides_unroll = strides*len(imgs)
        pads_unroll = pads*len(imgs)
        kdims_unroll = kdims*len(imgs)

        local_ifm_stats, local_ifm_wgt_stats, local_ofm_stats = p.starmap(_local_extract, [(imgs_unroll[i], layers_unroll[i], strides_unroll[i], pads_unroll[i], kdims_unroll[i]) for i in range(len(imgs)*len(layers))])

    print(local_ifm_stats.shape)
                
    total_stats = {'ifm_sum':      ifm_stats['sum'],
                   'ifm_mean':     ifm_stats['mean'],
                   'ifm_std':      ifm_stats['std'],
                   'ifm_absmax':   ifm_stats['absmax'],
                   'ifm_density':  ifm_stats['density'],
                   'ifm_channels': ifm_stats['channels'],
                   'ifm_shape':    ifm_stats['shape'],
                   'ifm_wgt_sum_sum': ifm_wgt_stats['sum_sum'],
                   'ifm_wgt_sum_mean': ifm_wgt_stats['sum_mean'],
                   'ifm_wgt_sum_std': ifm_wgt_stats['sum_std'],
                   'ifm_wgt_sum_absmax': ifm_wgt_stats['sum_absmax'],
                   'ifm_wgt_sum_density': ifm_wgt_stats['sum_density'],
                   'ifm_wgt_mean_sum': ifm_wgt_stats['mean_sum'],
                   'ifm_wgt_mean_mean': ifm_wgt_stats['mean_mean'],
                   'ifm_wgt_mean_std': ifm_wgt_stats['mean_std'],
                   'ifm_wgt_mean_absmax': ifm_wgt_stats['mean_absmax'],
                   'ifm_wgt_mean_density': ifm_wgt_stats['mean_density'],
                   'ifm_wgt_std_sum': ifm_wgt_stats['std_sum'],
                   'ifm_wgt_std_mean': ifm_wgt_stats['std_mean'],
                   'ifm_wgt_std_std': ifm_wgt_stats['std_std'],
                   'ifm_wgt_std_absmax': ifm_wgt_stats['std_absmax'],
                   'ifm_wgt_std_density': ifm_wgt_stats['std_density'],
                   'ifm_wgt_absmax_sum': ifm_wgt_stats['absmax_sum'],
                   'ifm_wgt_absmax_mean': ifm_wgt_stats['absmax_mean'],
                   'ifm_wgt_absmax_std': ifm_wgt_stats['absmax_std'],
                   'ifm_wgt_absmax_absmax': ifm_wgt_stats['absmax_absmax'],
                   'ifm_wgt_absmax_density': ifm_wgt_stats['absmax_density'],
                   'ifm_wgt_density_sum': ifm_wgt_stats['density_sum'],
                   'ifm_wgt_density_mean': ifm_wgt_stats['density_mean'],
                   'ifm_wgt_density_std': ifm_wgt_stats['density_std'],
                   'ifm_wgt_density_absmax': ifm_wgt_stats['density_absmax'],
                   'ifm_wgt_density_density': ifm_wgt_stats['density_density'],
                   'offset_sum':      ofm_stats['sum'],
                   'offset_mean':     ofm_stats['mean'],
                   'offset_std':      ofm_stats['std'],
                   'offset_absmax':   ofm_stats['absmax']}

    #for category in total_stats:
    #    print("{}:{}".format(category, len(total_stats[category])))
    df = pd.DataFrame(total_stats, columns=['ifm_sum', 'ifm_mean', 'ifm_std', 'ifm_absmax', 'ifm_density', 'ifm_channels', 'ifm_shape',
                                            'ifm_wgt_sum_sum', 'ifm_wgt_sum_mean', 'ifm_wgt_sum_std', 'ifm_wgt_sum_absmax', 'ifm_wgt_sum_density',
                                            'ifm_wgt_mean_sum', 'ifm_wgt_mean_mean', 'ifm_wgt_mean_std', 'ifm_wgt_mean_absmax', 'ifm_wgt_mean_density',
                                            'ifm_wgt_std_sum', 'ifm_wgt_std_mean', 'ifm_wgt_std_std', 'ifm_wgt_std_absmax', 'ifm_wgt_std_density',
                                            'ifm_wgt_absmax_sum', 'ifm_wgt_absmax_mean', 'ifm_wgt_absmax_std', 'ifm_wgt_absmax_absmax', 'ifm_wgt_absmax_density',
                                            'ifm_wgt_density_sum', 'ifm_wgt_density_mean', 'ifm_wgt_density_std', 'ifm_wgt_density_absmax', 'ifm_wgt_density_density',
                                            'offset_sum', 'offset_mean', 'offset_std', 'offset_absmax'])
    correlate_matrix = df.corr()

    for key in correlate_matrix:
        if not ("offset" in key):
            del correlate_matrix[key]

    # Display results
    print(correlate_matrix)
    sn.heatmap(correlate_matrix, annot=True, cmap=sn.color_palette("vlag"))
    plt.show()
            
if __name__ == '__main__':
    num_layers = count_layers()
    strides, pads, kdims = get_strides_pads_kdims()
    num_ifms, num_imgs = count_ifms_imgs()
    print([num_layers, num_imgs])
    
    single = ([0], [0])
    layer_wise = [None]*num_layers
    for l in range(num_layers):
        layer_wise[l] = (range(SAMPLES), [l])
    single_image = ([0], range(num_layers))
    total = (range(SAMPLES), range(num_layers))
    corr = (list(range(100)), list(range(num_layers)))
    
    inputs = [single, layer_wise, single_image, total, None, corr]
    print("**Running MODE {}**".format(MODE_DEFS[MODE]))
    if MODE == 4:
        run_weights_only(num_layers)
        exit(0)
    elif MODE == 1:
        for i in inputs[MODE]:
            summary(i[0], i[1])
    elif MODE == 5:
        correlate_finegrain(inputs[MODE][0], inputs[MODE][1], strides, pads, kdims)
    else:
        summary(inputs[MODE][0], inputs[MODE][1])
