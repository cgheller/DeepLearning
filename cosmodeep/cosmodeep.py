#import all necessary libraries

import numpy as np
import tensorflow as tf
from util import *
import time
####sess = tf.InteractiveSession()
np.set_printoptions(threshold=np.inf)

# define some support functions to initalize weights and bias:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# define some functions to keep code clean:

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# START WORKING

readdata = 1
writefile = 1
checkpoint = 1
checkpointfile = './model-22.ckpt'
restart = 0
restartfile = './model-22.ckpt'

# *** INPUT LAYER ***

if readdata == 1:
   print("Loading data")

   path = '/scratch/snx3000/cgheller/CosmoDeep/results/log-flattenlow22/'
   #path = '/scratch/snx3000/cgheller/CosmoDeep/results/linear/'
   print("Reading: ")
   fullpath = path + 'X_processed.h5'
   print(fullpath)
   imagedata = load_hdf5_matrix(path + 'X_processed.h5')
   print("Reading: ")
   fullpath = path + 'y_processed.h5'
   print(fullpath)
   actualdata = load_hdf5_matrix(path + 'y_processed.h5')

   shap = actualdata.shape
   datasize = shap[0]
   print("total data size = ")
   print(datasize) 

   print("Data load completed")


# SETUP

print("Setting up")

# *** Set image and network parameters and variables ***

# Training set source size
bsize = 9800
# Input image resolution
imagex = 160
imagey = 160
imagesize = imagex*imagey
# number of neurons in the third hidden layer (fully connected)
Nneurons = 1024
# number of neurons in the output layer
Nout = 2
# convolution window size
Wx = 3
Wy = 3
# number of input channel
Nin = 1
# number of feature maps in hidden layer 1
Nch = 32
# number of feature maps in hidden layer 2
Nch2 = 64
# Learning Rate
eta = 1e-5
# Mini Batch size
nbatch = 50
# Total number of iterations
nstep = 350
# Number of maxpool layers
Nmaxpool = 2

# *** FIRST HIDDEN LAYER ***

# Initialize weights and bias using the support fuctions defined above for the first hidden layer
W_conv1 = weight_variable([Wx, Wy, Nin, Nch])
b_conv1 = bias_variable([Nch])

# create necessary arrays and placeholders (to be used by tensorflow)
# input data placeholder
x = tf.placeholder(tf.float32, [None, imagesize])
xaux = tf.placeholder(tf.float32, [None, imagesize])
# reshape x to a 4D tensor (needed for convolution step)
x_image = tf.reshape(x, [-1,imagex,imagey,Nin])

# define the first hidden layer (ReLU convolution + max_pool)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# *** SECOND HIDDEN LAYER ***

# define the second hidden layer (ReLU convolution + max_pool) NOTE CHANGES IN THE NUMBER OF CHANNELS!

W_conv2 = weight_variable([Wx, Wy, Nch, Nch2])
b_conv2 = bias_variable([Nch2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Feature map reduced to 1/4 * 1/4 (due to maxpool layers)

featx = imagex/(Nmaxpool*Nmaxpool)
featy = imagey/(Nmaxpool*Nmaxpool)

# *** THIRD HIDDEN LAYER ***

# Fully connected layer (1024 ReLU Neurons)
# weights (initialized with previous function):
W_fc1 = weight_variable([int(featx * featy * Nch2), Nneurons])
# biases (initialized with previous function):
b_fc1 = bias_variable([Nneurons])

# Flatten the tensor
h_pool2_flat = tf.reshape(h_pool2, [-1, int(featx * featy * Nch2)])
# define the fully connected network
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout to reduce overfitting

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# *** OUTPUT LAYER ***

# Fully connected Softmax layer for final classification

W_fc2 = weight_variable([Nneurons, Nout])
b_fc2 = bias_variable([Nout])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# *** DEFINE REMAINING NETWORK ELEMENTS ***
# 1: placeholder for the correct answer (for training)
ycorrect = tf.placeholder(tf.float32, [None, Nout])

# 2: define the cost function (cross_entropy)
#original
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ycorrect * tf.log(y_conv), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ycorrect, logits=y_conv))

# 3: instance backpropagation with learning rate eta, using ADAM 
train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy)

# 4: compare to correct result

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ycorrect,1))
correct_prediction1 = y_conv
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# *** RUN THE TRAINING LOOP ***

start_time = time.time()
saver = tf.train.Saver()
#
print("Training the network")
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(init)
if restart == 1:
   print("Restarting system")
   load_path = saver.restore(sess, restartfile)
#if (nstep*nbatch) > (datasize-nbatch):
#    print("Number of iterations too large -> Exiting...")
#    exit()
#print(np.amin(imagedata[0,:]))
#print(np.amax(imagedata[0,:]))
#print(np.amin(imagedata[1,:]))
#print(np.amax(imagedata[1,:]))

imagedataR = np.zeros((nbatch,imagex*imagey))
actualdataR = np.zeros((nbatch,Nout))

for i in range(nstep):
    ###############################################batch = mnist.train.next_batch(nbatch)
    #print(sess.run(signalnoise, feed_dict={x:batch[0]}))
    ###datainput = sess.run(signalnoise, feed_dict={x:batch[0]})
    #print(datainput)

    ##### writing a jpeg file
    #####f = open("test.jpeg", "wb+")
    #datainput256 = tf.mul(datainput, 255)
    #####datainput256 = tf.mul(imagedata[1,:], 255)
    #####datainputuintaux = tf.cast(datainput256, tf.uint8)
    #####datainputuint = tf.reshape(datainputuintaux,[imagex,imagey,Nin])
    #####datajpg = tf.image.encode_jpeg(datainputuint)
    #####f.write(sess.run(datajpg))
    #####f.close()

# Extract a random batch

    idx = np.arange(bsize)
    rdm = np.random.RandomState(None)#(i)
    rdm.shuffle(idx)

# Batch extrema
    istart = 0
    iend = nbatch-1

    for j in range(nbatch):
        imagedataR[j,:] = imagedata[idx[j], :]
        actualdataR[j,:] = actualdata[idx[j], :]

    if i%50 == 0:
       train_accuracy = sess.run(accuracy, feed_dict={x: imagedataR[istart:iend,:], ycorrect: actualdataR[istart:iend,:], keep_prob:1.0})
       print("step %d, training accuracy %g"%(i, train_accuracy))
    #print(istart,iend)
    sess.run(train_step, feed_dict={x: imagedataR[istart:iend,:], ycorrect: actualdataR[istart:iend,:], keep_prob:0.75})

# Save session

if checkpoint == 1:
    save_path = saver.save(sess, restartfile)
    print('Model saved in %s' % save_path)



# *** COMPUTE ACCURACY ON THE TEST IMAGES ***

print("Calculating Accuracy")
###nbatch1=10000
# clean-noise:
istart = bsize+1000
iend = bsize+2000
# strong-noise
#istart = 9000
#iend = 10000
###datainput1 = sess.run(signalnoise1, feed_dict={x:imagedata[istart:iend,:]})
###for j in range(imagesize):
###    print(datainput1[2][j])


## writing jpeg files
if writefile == 1:
 #for i in range(iend-istart):
 for i in range(50):
  if actualdata[istart+i,0] == 1:
    filename = "gal"+str(i)+".jpg"
  if actualdata[istart+i,1] == 1:
    filename = "web"+str(i)+".jpg"
#  if actualdata[istart+i,2] == 1:
#    filename = "spike"+str(i)+".jpg"
  f = open(filename, "wb+")
  xmax = np.amax(imagedata[istart+i,:])
  xmin = np.amin(imagedata[istart+i,:])
  xmean = np.mean(imagedata[istart+i,:])
  xstd = np.std(imagedata[istart+i,:])
  iaux=bsize+i
  #print(iaux, xmax, xmean, xmax/xmean)
  datainput256 = tf.multiply(imagedata[istart+i,:], 255)
  datainputuintaux = tf.cast(datainput256, tf.uint8)
  datainputuint = tf.reshape(datainputuintaux,[imagex,imagey,Nin])
  datajpg = tf.image.encode_jpeg(datainputuint)
  f.write(sess.run(datajpg))
  f.close()


print(sess.run(correct_prediction1, feed_dict={x: imagedata[istart:iend,:], ycorrect: actualdata[istart:iend,:], keep_prob: 1.0}))
test_acc = sess.run(accuracy, feed_dict={x: imagedata[istart:iend,:], ycorrect: actualdata[istart:iend,:], keep_prob: 1.0})
print("test accuracy %g"%test_acc)

elapsed_time = time.time() - start_time
print(' ')
print(elapsed_time)

sess.close()
