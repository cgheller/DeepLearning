import numpy as np
import pickle
from util import *
import gc
from os.path import join
import os
import pyfits
##from skimage import exposure
import time

selection_rate=10

def log10(x):
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    for i in range(x.shape[0]):
     for j in range(x.shape[1]):
           numerator = tf.log(x[i][j])
           x[i][j] = numerator / denominator
    return x

def flatten(x):
    minconst=float(1e-27)
    maxconst=float(1e-10)
    for i in range(x.shape[0]):
     for j in range(x.shape[1]):
        if x[i][j] <= minconst:
           x[i][j] = minconst
        if x[i][j] >= maxconst:
           x[i][j] = maxconst
    return x    

def flattenlow(x):
    minconst=float(1e-20)
    for i in range(x.shape[0]):
     for j in range(x.shape[1]):
        if x[i][j] <= minconst:
           #x[i][j] = minconst
           x[i][j] = 1e-27
    return x


def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening


#different preprocessing techiques
#it is important that the data is in a range between 0 and 1 or -1 and 1
#otherwise traning can be slow and the model might diverge during traning

def std_norm_whole_dataset(X):
    print(np.mean(X,0).shape)
    X -=np.mean(X,0)
    X/= np.std(X,0)
    
    print(np.mean(X,0))
    print(np.std(X,0))
    return X

#this procedure offered the best generalization performance for the filament data
def max_norm_whole_dataset(X):
    print(np.mean(X,0).shape)
    X -=np.min(X,0)
    X/= np.max(X,0)
    
    print(np.min(X,0))
    print(np.max(X,0))
    return X

def max_norm_per_image_dataset(X):
    for i in range(X.shape[0]):
        img = X[i]
        img -= np.min(img)
        if np.max(img) > 0.0:
            img = img/np.max(img)
        X[i] = img
    return X

def std_norm_per_image_dataset(X):
    for i in range(X.shape[0]):
        img = X[i]
        img -= np.mean(img)
        img/= np.std(img)
        X[i] = img
    return X


'''
X1 = X1 - np.min(X1)
X1 = X1 / np.max(X1,0)

X2 = X2 - np.min(X2)
X2 = X2 / np.max(X2,0)
'''

# Set all the important paths (not all used)

mypath    = '/scratch/snx3000/cgheller/CosmoDeep/data/'
outpath   = '/scratch/snx3000/cgheller/CosmoDeep/results/noise5.0-tot/'

# Set the number of classes to classify

numberofclasses = 2

# Set data sources sub folders
folder_names = ['tot','noise']
#########################folder_names = ['out/clean','out/noise']

def get_data(mypath,nnn):
    t0 = time.time()
    n = 10000
    data = []
    labels = []
    paths = []
    totcount=0
    totcount2=0
    if nnn == 1:
        nstart=0
    else:
        nstart=50000

    for i in range(n):
        path = join(mypath, str(i))+'.fits'
        noisepath = mypath+'/../noise/'+str(nstart+i)+'.fits'

        if not os.path.exists(path): continue

        # Reading data from FITS files
        if nnn == 1:
          img = pyfits.getdata(path,0,memmap=False)           
          noise = pyfits.getdata(noisepath,0,memmap=False)
          noise = 5.0*noise
          img = np.add(img,noise)
          print(path,noisepath)
        else:
          img = pyfits.getdata(noisepath,0,memmap=False)
          print(noisepath)

        # Normalize data
        # log scale
        #img_adapteq = flattenlow(img)
        img_adapteq = np.fabs(img)
        img_adapteq = np.log10(img_adapteq)
        # linear scale
        #img_adapteq = img
        # flattened linear scale
        #img_adapteq = flatten(img)
        xmax = np.amax(img_adapteq)
        xmin = np.amin(img_adapteq)

        img_adapteq = img_adapteq-xmin
        if xmax-xmin == 0 : 
           print("---------------------------> SKIPPED")
           continue
        img_adapteq = img_adapteq/(xmax-xmin)

        xmin = np.amin(img_adapteq)
        xmean = np.mean(img_adapteq)#+5*xstd
        xstd = np.std(img_adapteq)
        xmax = np.amax(img_adapteq)
        print(xmean,xstd,xmin,xmax)
        
        label = 1
        totcount=totcount+1
        #this is the preprocessing algorithm, comment this out to remove the preprocessing of the image completely
        ###img_adapteq = exposure.equalize_adapthist(np.log(img_adapteq + 1.0), clip_limit=0.5,kernel_size=(4,4))

        #saving the paths is useful to restore which array belonged to which image on the harddrive
        paths.append(path)
        #add data to list
        data.append(img_adapteq)
        #
        labels.append(label)
    #
    print(totcount)
    #convert list to matrix to later save this as hdf5 file
    return [np.array(data,dtype=np.float32), np.array(paths), np.array(labels)]


# data with signal: X1 is the array of pixels, P1 the corresponding filename
print("Starting data load")
X1, P1, L1 = get_data(join(mypath, folder_names[0]),1)
print("Web Loaded")

# data with pure noise: X2 is the array of pixels, P2 the corresponding filenament
X2, P2, L2 = get_data(join(mypath, folder_names[1]),2)
print("Noise Loaded")

# gc is the garbage collector (automatic memory management)
gc.collect()

# reshape of the filenames array: to be clarified why...
P1 = P1.reshape(-1,1)
P2 = P2.reshape(-1,1)

gc.collect()

#these are the labels 0 for one class, 1 for the other class
y1 = np.zeros((X1.shape[0],numberofclasses))
y2 = np.zeros((X2.shape[0],numberofclasses))
for i in range (X1.shape[0]):
    ####if L1[i] == 1:
    ####>if L1[i] == 2:
    y1[i,1] = 1
for i in range (X2.shape[0]):
    y2[i,0] = 1

##flip images by 90 degree intervals (4 possible flips), use both signal and corresponding noise
#X = np.vstack([X1, np.fliplr(X1), X2, np.fliplr(X2)])
##X = np.vstack([X1, X1, X2, X2])
#P = np.vstack([P1, P1, P2, P2])
#print("Images flipped")
X = np.vstack([X1, X2])
P = np.vstack([P1, P2])

del X1
del X2
gc.collect()
#X = np.vstack([X1,X2])
#print(X[0].sum())
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
#print(X[0].sum())

##y = np.vstack([y1, y1, y2, y2])
y = np.vstack([y1,y2])


#randomize the data
idx = np.arange(X.shape[0])
rdm = np.random.RandomState(1234)
rdm.shuffle(idx)

gc.collect()
X = X[idx]
P = P[idx]
y = y[idx]

#print(P[0:7])
#print(P)

gc.collect()

###X = std_norm_per_image_dataset(X)
###############################X = max_norm_per_image_dataset(X)
###X = max_norm_whole_dataset(X)

print("Writing data")
#save as hdf5 (from util file)
save_hdf5_matrix(outpath + 'X_processed.h5', np.float32(X))
save_hdf5_matrix(outpath + 'y_processed.h5', np.float32(y))
save_hdf5_matrix(outpath + 'idx.h5', idx)
#pickle.dump(P, open(join(outpath, 'P.p'), 'w'))
#i#############################################print(P[50000:51000])
