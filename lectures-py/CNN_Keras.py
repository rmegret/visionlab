
# coding: utf-8



# <markdowncell>
# # Neural Networks using Keras


# <markdowncell>
# - Most *object detection/labeling/segmentation/classification* tasks now have
#   neural network equivalent algorithms that perform on-par with or better than
#   hand-crafted methods.
# - One library that gives Python users particularly easy access to deep
# learning is Keras: https://github.com/fchollet/keras/tree/master/examples (it
# works with both Theano and TensorFlow).


# <markdowncell>
# ### Configurations
# 
# From http://www.asimovinstitute.org/neural-network-zoo/:
# 
# <img src="neuralnetworks.png" style="width: 80%"/>


# <markdowncell>
# ### Preliminary: Installing Keras
# 
# Generic instructions to install Keras with TensorFlow can be found here:
# - https://keras.io/#installation
# - https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-
# backend/
# 
# If you use Anaconda, you can replace the `pip` installer with `conda`,
# although both should work. You should already have: `numpy scipy scikit-learn
# scikit-image pillow h5py` installed. Keras and tensorflow are available from
# conda-forge for the major systems/OS:
# ```
# conda install -c conda-forge tensorflow
# conda install -c conda-forge keras
# conda install -c conda-forge graphviz pydot
# ```
# 
# Test if installation was successfull in python:
# ```
# Python 3.5.1 |Continuum Analytics, Inc.| (default, Jun 15 2016, 16:14:02)
# ...
# 
# In [1]: import keras
# Using TensorFlow backend.
# W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library
# wasn't compiled to use SSE4.1 instructions, but these are available on your
# machine and could speed up CPU computations.
# ...
# In [2]:
# ```
# You can ignore such warnings, these simply indicate that the base library for
# tensorflow was installed, which will work just fine, maybe not as fast as it
# could be. You may want to install the GPU version `tensorflow-gpu`, much
# faster, but will require some tuning specific to your system. If a GPU version
# is installed, Keras should normally use it automatically.
# - https://www.tensorflow.org/install/
# - MacOS support has been dropped recently, workaround:
# https://stackoverflow.com/questions/44744737/tensorflow-mac-os-gpu-support
# - Windows: http://inmachineswetrust.com/posts/deep-learning-setup/ and
# https://blog.paperspace.com/running-tensorflow-on-windows/
# 
# 
# In the Keras docs, you may read about `image_data_format`.  By default, this
# is `channels-last`, which is
# compatible with scikit-image's storage of `(row, cols, ch)` and the most
# efficient when using TensorFlow. Check your config `~/.keras/keras.json`,
# which should look like:
# ```
# {
#     "image_data_format": "channels_last",
#     "epsilon": 1e-07,
#     "backend": "tensorflow",
#     "floatx": "float32"
# }
# ```


# <codecell>

# Test your installation:
import tensorflow
import keras
import pydot   


# <codecell>

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline   


# <markdowncell>
# ## Keras basics
# 
# In this section, we apply a simple NN to simple 2D data to introduce the basic
# elements of Keras:
# - Creating a neural-network model
# - Training it and evaluating the accuracy
# - Plotting the learning curves


# <codecell>

## Generate dummy data with some structure
from sklearn import datasets
from sklearn.model_selection import train_test_split

# X, y = datasets.make_classification(n_features=2, n_samples=200, n_redundant=0, n_informative=1,
#                                     n_clusters_per_class=1, random_state=42, class_sep=1.0)

X, y = datasets.make_moons(n_samples=500, noise=0.25)
X[:,0]-=0.5 # Center the moons

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#X_test[:,1]+=15

plt.scatter(X[:,0],X[:,1], c=y)
#plt.plot(X_train[:,0],X_train[:,1], 'r.', markersize=1)
#plt.plot(X_test[:,0],X_test[:,1], 'b.', markersize=1)   


# <codecell>

# Create a Neural Network with two hidden layers

from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer

N=64

model = Sequential()
model.add(InputLayer(input_shape=[2]))
model.add(Dense(N, activation='relu'))
model.add(Dense(N, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
#model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])   


# <codecell>

model.summary(line_length=100)

# Visualize the network
from IPython.display import SVG, Image
from keras.utils.vis_utils import model_to_dot # need pydot and graphviz packages

Image(model_to_dot(model,rankdir='LR',show_layer_names=False, show_shapes=False).create(prog='dot', format='png'))
#SVG(model_to_dot(model,rankdir='LR',show_layer_names=False).create(prog='dot', format='svg'))
#plot_model(model, to_file='model.png') # Save to an image file   


# <codecell>

# Train the network
epochs=100
history = model.fit(X_train, y_train, epochs=epochs,
          batch_size=100, validation_data=(X_test,y_test),verbose=1)
score = model.evaluate(X_test, y_test)

print('\n\nAccuracy:', score[1])
score   


# <markdowncell>
# ### Plotting learning curve


# <codecell>

def plot_history(history):
    epochs = len(history.history['acc'])
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    ticks=range(1,epochs+1)
    axes[0].plot(ticks,history.history['acc'], label='train')
    axes[0].plot(ticks,history.history['val_acc'], label='test')
    axes[0].legend()
    axes[0].set_title('accuracy')
    axes[0].set_xlabel('epochs')
    axes[1].plot(ticks,history.history['loss'], label='train')
    axes[1].plot(ticks,history.history['val_loss'], label='test')
    axes[1].legend()
    axes[1].set_title('loss')
    axes[1].set_xlabel('epochs')
    #plt.xticks(range(0,nepochs,5));   


# <codecell>

plot_history(history)   


# <markdowncell>
# For simple problems, classifiers such as random forests can actually provide
# same or better performance faster than neural networks. Let's see with more
# challenging problem in next section.


# <markdowncell>
# ### Debugging the decision function (in 2D)


# <codecell>

# Visualize the decision function

def plot_2d_decision(model, X, y):
    assert X.shape[1]==2, 'X should have 2 columns, got {}'.format(X.shape[1])
    
    # Generate a 2D grid of input samples
    B=np.max(X)*1.1
    gridy, gridx = np.mgrid[-B:B:100j,-B:B:100j]
    X_grid = np.concatenate((gridx.reshape(-1,1),gridy.reshape(-1,1)),axis=1)

    # Apply the network to this input
    out = model.predict(X_grid)
    out_im = out.reshape(gridx.shape)
    
    # Visualize the result overlayed with the dataset
    plt.imshow(out_im, extent=(-B,B,B,-B), cmap='gray')
    plt.contour(out_im, levels=[0.5], colors=['r'], extent=(-B,B,-B,B))
    plt.scatter(X[:,0],X[:,1], c=y, marker='.')   


# <codecell>

plot_2d_decision(model, X, y)   


# <markdowncell>
# ### Evaluating the meta-parameters (number of nodes, layers...)


# <codecell>

# Function that evaluates a network created according to meta-parameters 
def evaluate_perf(N=64, hidden=2, epochs=50):
    print('evaluate_perf: N={}, hidden={}...'.format(N, hidden))
    
    model = Sequential()
    model.add(InputLayer(input_shape=[2]))
    for i in range(hidden):
        model.add(Dense(N, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs,
              batch_size=100, verbose=0, validation_data=(X_test, y_test))
    loss, acc = model.evaluate(X_test, y_test)
    
    print('  accuracy={:.3f}'.format(acc))
    plot_2d_decision(model, X, y)
    
    plt.title('N={}, h={}: acc={:.3f}'.format(N, hidden, acc))
    
    perf = {'acc':acc, 'history':history}
    
    return perf   


# <codecell>

# Illustration of over-fitting: too many parameters
perf = evaluate_perf(N=256, hidden=4, epochs=200)
fig = plt.figure()
plot_history(perf['history']);   


# <codecell>

params=[(16,0),(16,1),(16,2),(16,3),
        (64,0),(64,1),(64,2),(64,3)]
nx=4
ny=(len(params)+nx-1)//nx

acc_array = np.zeros(len(params))
fig,axes = plt.subplots(ny,nx,figsize=(12,7))
axes=axes.ravel()
for i,(N,h) in enumerate(params):
    plt.sca(axes[i])
    perf = evaluate_perf(N,h, 100)
    acc_array[i] = perf['acc']   


# <codecell>

fig2 = plt.figure(figsize=(8,4))
plt.bar(range(len(params)),acc_array, tick_label=[str(p) for p in params])
plt.ylabel('accuracy')
plt.xlabel('(nb_units,nb_layers)')   


# <markdowncell>
# ### Comparing to `sklearn` classifiers


# <codecell>

def plot_history_compare(history, acc_dict):
    """
    Plot NN fit history (test accuracy) and overlay fixed accuracy from other classifier
    
    history: output of model.fit()
    acc_dict: dictionnary of the form {'approach_name': accuracy, ...}
    """
    epochs = len(history.history['acc'])
    
    plt.plot([],[],label='_nolegend_')
    
    ticks=range(1,epochs+1)
    plt.plot(ticks,history.history['val_acc'], label='test')
    plt.legend()
    plt.title('accuracy')
    plt.xlabel('epochs')
    
    epochs = len(history.history['acc'])
    for label in acc_dict:
        acc = acc_dict[label]
        plt.gcf().axes[0].plot([0,epochs],[acc,acc], ":")
    plt.gcf().axes[0].legend(['NN test'] + list(acc_dict.keys()))   


# <codecell>

# Compare to RandomForest and SVM

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import OrderedDict

perfs=OrderedDict()

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
perfs['SVM-Lin'] = svm.score(X_test, y_test)

svm = SVC()
svm.fit(X_train, y_train)
perfs['SVM-RBF'] = svm.score(X_test, y_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
perfs['RF'] = rf.score(X_test, y_test)

plot_history_compare(history, perfs)   


# <markdowncell>
# ## Simple CNN for object recognition
# 
# When dealing with images, Convolutional Neural Network offer the advantage of
# computing visual features from the image using Convolutional Layers, which can
# be trained at the same time as the dense layers that perform the
# classification.
# 
# In this section, we use the Fashion-MNIST dataset to recognize pieces of
# clothes shown as 28x28 pixel gray-scale images.
# 
# Dataset from https://github.com/zalandoresearch/fashion-mnist. The training
# set contains 60,000 samples and the test set 10,000 samples. Each training and
# test example is assigned to one of the following labels:
# ```
#     0 T-shirt/top
#     1 Trouser
#     2 Pullover
#     3 Dress
#     4 Coat
#     5 Sandal
#     6 Shirt
#     7 Sneaker
#     8 Bag
#     9 Ankle boot
# ```


# <codecell>

from keras.datasets import fashion_mnist, mnist
from keras.utils import np_utils

data=fashion_mnist.load_data()
# Download will be done only once:
# By default, the dataset will be cached inside ~/.keras/datasets/

#data=mnist.load_data() # Uncomment to use original MNIST instead

((X_train,y_train),(X_test,y_test))=data

# To go faster, reduce the amount of training data
X_train=X_train[:2500]
y_train=y_train[:2500]

# Prepare datasets
# This step contains normalization and reshaping of input.
# For output, it is important to change number to one-hot vector. 
X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)   


# <codecell>

fig,axes = plt.subplots(4,10, figsize=(12,4))
axes=axes.ravel()
for i in range(len(axes)):
    axes[i].imshow(X_train[i,0,:,:], cmap='gray')
    axes[i].set_xticks([]); axes[i].set_yticks([])
#a.set_xtick(None)   


# <codecell>

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer, BatchNormalization, Conv2D, MaxPool2D, Flatten
from keras.initializers import Constant

model = Sequential()
model.add(InputLayer(input_shape=(1, 28, 28)))
model.add(BatchNormalization())
model.add(Conv2D(32, (2, 2), 
        padding='same', 
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform'
    ))
model.add(MaxPool2D(padding='same'))
model.add(Conv2D(32, (2, 2), 
        padding='same', 
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform', 
        input_shape=(1, 28, 28)
    ))
model.add(MaxPool2D(padding='same'))
model.add(Flatten())
model.add(Dense(128,
        activation='relu',
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform',         
    ))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)   


# <codecell>

model.summary()

from IPython.display import SVG, Image
from keras.utils.vis_utils import model_to_dot # need pydot and graphviz packages

Image(model_to_dot(model,rankdir='LR',show_layer_names=False).create(prog='dot', format='png'))   


# <codecell>

history = model.fit(
    X_train, 
    y_train, 
    epochs=20, 
    batch_size=100, 
    validation_data=(X_test, y_test),
    verbose=1,
)
model.evaluate(X_test,y_test)   


# <codecell>

plot_history(history)   


# <codecell>

# Compare to RandomForest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train.reshape((X_train.shape[0],-1)), y_train)
acc=rf.score(X_test.reshape((X_test.shape[0],-1)), y_test)

plot_history_compare(history, {'RF':acc})   


# <markdowncell>
# It is clear that CNN outperform RF on this dataset. Their architecture is
# better at extracting the relevant information from the image compared to
# passing the raw image data to the random forest. Improving the performance of
# the RF would require designing better features to be fed to the RF classifier,
# which the CNN includes in the first layers.


# <markdowncell>
# ## Object recognition using Inception-v3
# 
# In this section, we use the pre-trained model Inception-v3 for object
# recognition.
# 
# <img width="600px" src="inception_v3_architecture.png"/>
# 


# <codecell>

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Loading the pre-trained model 
# (takes a few seconds, may need downloading the first time)
net = InceptionV3()   


# <codecell>

if False: # Change to true to display Inception structure (very large)
    net.summary(line_length=100, positions=[0.35,0.65,0.7,1.0])
    Image(model_to_dot(net,rankdir='TB',show_layer_names=False).create(prog='dot', format='png'))   


# <codecell>

from skimage import transform

def inception_predict(image):
    # Rescale image to 299x299, as required by InceptionV3
    image_prep = transform.resize(image, (299, 299, 3), mode='reflect')
    
    # Scale image values to [-1, 1], as required by InceptionV3
    image_prep = (img_as_float(image_prep) - 0.5) * 2
    
    predictions = decode_predictions(
        net.predict(image_prep[None, ...])
    )
    
    plt.imshow(image, cmap='gray')
    
    for pred in predictions[0]:
        (n, klass, prob) = pred
        print('{klass:>15} ({prob:.3f})'.format(klass=klass, prob=prob))   


# <codecell>

from skimage import data, img_as_float
inception_predict(data.chelsea())   


# <codecell>

inception_predict(data.camera())   


# <codecell>

inception_predict(data.coffee())   


# <codecell>

inception_predict(data.stereo_motorcycle()[1])   


# <markdowncell>
# You can fine-tune Inception to classify your own classes, as described at
# 
# https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes


# <markdowncell>
# ---


# <markdowncell>
# Extra topics:
# - Many examples provided with Keras (ResNet, OCR, AutoEncoders, ...):
# https://github.com/fchollet/keras/tree/master/examples
# - Using Keras as a Scikit-learn classifier: https://keras.io/scikit-learn-api/


# <markdowncell>
# ### ResNet for object recognition
# 
# We can also try ResNet, which is included in Keras distribution.


# <codecell>

from keras.applications.resnet50 import ResNet50
from keras.applications import resnet50

# Loading the pre-trained model 
# (may take a few minutes to download for the first time)
resnet = ResNet50()   


# <codecell>

if False:
    resnet.summary()
    Image(model_to_dot(resnet,rankdir='TB',show_layer_names=False).create(prog='dot', format='png'))   


# <codecell>

def resnet_predict(image):
    # Rescale image to 224x224, as required by ResNet50
    image_prep = transform.resize(image, (224, 224, 3), mode='reflect')
    
    # Scale image values to [-128, 128]
    image_prep = img_as_float(image_prep)*256 - 128
    
    predictions = resnet50.decode_predictions(
        resnet.predict(image_prep[None, ...])
    )
    
    plt.imshow(image, cmap='gray')
    
    for pred in predictions[0]:
        (n, klass, prob) = pred
        print('{klass:>15} ({prob:.3f})'.format(klass=klass, prob=prob))   


# <codecell>

resnet_predict(data.chelsea())   


# <codecell>

resnet_predict(data.camera())   


# <codecell>

resnet_predict(data.coffee())   


# <codecell>

resnet_predict(data.stereo_motorcycle()[1])   


# <codecell>

resnet_predict(data.rocket())   
