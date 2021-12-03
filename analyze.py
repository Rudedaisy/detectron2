import numpy as np
import os

import matplotlib.pyplot as plt

PATH = "/home/rudedaisy/detectron2/export/"
f_log = "log"
f_model = "model.csv"


# Count the number of offset maps in the model
num_layers = 0
f = open(os.path.join(PATH, f_model), "r")
for line in f:
    if "conv2_offset" in line:
        num_layers += 1
f.close
        
# Count the number of images processed
num_ifms = 0
f = open(os.path.join(PATH, f_log), "r")
for line in f:
    if len(line) > 2:
        num_ifms += 1
f.close()
assert num_ifms % num_layers == 0
num_imgs = num_ifms // num_layers

print([num_layers, num_imgs])


def statLayer(fileName):
    ofm = np.load(fileName)
    print(ofm.shape)

    # Need to check a histogram to see if distribution is normal...
    #buckets = []
    #bucketIDs = []
    #for i in np.arange(-1,1,0.1):
    #    buckets.append(0)
    #    bucketIDs.append(i)
    size = len(np.arange(-1,1,0.1))
    #for val in ofm.flatten():
    #    for i in range(size):
    #        if val >= bucketIDs[i] and ((i < size-1) or val < bucketIDs[i]):
    #            buckets[i] += 1
    #            continue
    #print(buckets)
    plt.hist(ofm.flatten(), density=True, bins = size)
    plt.ylabel('Probability')
    plt.xlabel('Offset')
    plt.show()
    
    #mean = np.mean(ofm, axis=(0,2,3))
    mean = np.mean(ofm)
    std = np.std(ofm)
    print(mean)
    print(std)
    exit(0)

for i in range(num_imgs):
    for l in range(num_layers):
        statLayer(os.path.join(PATH, "OFM-"+str(i)+"-"+str(l)+".npy"))
