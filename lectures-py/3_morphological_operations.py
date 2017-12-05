
# coding: utf-8



# <markdowncell>
# # Morphological operations
# 
# Morphology is the study of shapes. In image processing, some simple operations
# can get you a long way. In this lab, we discuss the following morphological
# operations:
# - Erosion, Dilation
# - Opening, Closing
# 
# The detailed documentation for scikit-image's morphology module is
# [here](http://scikit-image.org/docs/0.10.x/api/skimage.morphology.html).


# <codecell>

# IMPORTS
from __future__ import division, print_function
%matplotlib inline

import numpy as np
from matplotlib import pyplot as plt, cm
import skdemo
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'
from skimage import morphology   


# <codecell>

# image = np.array([[0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 1, 1, 1, 0, 0],
#                   [0, 0, 1, 1, 1, 0, 0],
#                   [0, 0, 1, 1, 1, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

from skimage import io
image = io.imread('../images/bwshapes.png')

plt.imshow(image);   


# <markdowncell>
# ## Structuring element
# Importantly, we must use a *structuring element*, which defines the local
# neighborhood of each pixel. To get every neighbor (up, down, left, right, and
# diagonals), use `morphology.square`; to avoid diagonals, use
# `morphology.diamond`:


# <codecell>

from skimage import morphology
sq = morphology.square(width=7)
dia = morphology.diamond(radius=3)
print("sq=",sq)
print("dia=",dia)

def show_strel(axes,im):
    h=im.shape[0]; w=im.shape[1];
    h2=h/2; w2=w/2;
    axes.imshow(im, vmin=0, vmax=1, extent=(-w2,w2,-h2,h2))
    axes.plot(0,0,'+b', scalex=False, scaley=False)
    axes.set_xticks(np.arange(w)-w2+0.5)
    axes.set_yticks(np.arange(h)-h2+0.5)

fig, axes = plt.subplots(1,2)
show_strel(axes[0],sq)
show_strel(axes[1],dia)   


# <markdowncell>
# The center of the structuring element represents the pixel being considered,
# and the surrounding values are the neighbors: a 1 value means that pixel
# counts as a neighbor, while a 0 value does not. Note that most structuring
# elements have odd width and height (3, 5, 7...) in order to be symmetrical
# with respect to the central pixel.


# <markdowncell>
# ## Erosion and Dilation
# 
# The first things to learn are *erosion* and *dilation*. In erosion, we look at
# a pixel’s local neighborhood and replace the value of that pixel with the
# minimum value of that neighborhood. In dilation, we instead choose the
# maximum.


# <markdowncell>
# ### Erosion
# Erosion by the square structuring element produces:


# <codecell>

sq_erode = morphology.erosion(image, sq)
skdemo.imshow_all(image, sq_erode, shape=(1, 2))   


# <markdowncell>
# **Discussion:** What are the effects of the erosion
# on the various elements of the image ?
# 
# - ...
# - ...


# <markdowncell>
# ### Dilation
# Dilation by the square produces:


# <codecell>

sq_dilate = morphology.dilation(image, sq)
skdemo.imshow_all(image, sq_dilate)   


# <markdowncell>
# **Discussion:** What are the effects of the dilation on the various elements
# of the image ?
# - ...
# - ...


# <markdowncell>
# ### Shape of the structuring element
# 
# If we dilate by a diamond structurating element, we obtain instead:


# <codecell>

skdemo.imshow_all(image, morphology.dilation(image, sq), morphology.dilation(image, dia),
                 titles=['orig','strel=square','strel=diamond'] )   


# <markdowncell>
# **Discussion:** what are the differences between dilation by a square and
# dilation by a diamond?
# - ...
# - ...


# <markdowncell>
# ## Opening and Closing
# Erosion and dilation can be combined into two slightly more sophisticated
# operations, *opening* and *closing*.


# <markdowncell>
# What happens when run an erosion followed by a dilation of this image?
# 
# What about the reverse?
# 
# Try to imagine the operations in your head before trying them out below.


# <codecell>

sq_open = morphology.opening(image, sq)
skdemo.imshow_all(image, sq_erode, sq_open, titles=['original','erode','erode then dilate = open']) # erosion -> dilation   


# <codecell>

sq_close = morphology.closing(image, sq)
skdemo.imshow_all(image, sq_dilate, sq_close, titles=['original','dilate','dilate then erode = close']) # dilation -> erosion   


# <markdowncell>
# ## Alternated filters
# 
# To regularize imperfections in both black and white, both opening and closing
# can be used in alternation


# <codecell>

sq_open_close = morphology.closing(sq_open, sq)

fig, axes = plt.subplots(1,3, figsize=(12,6))
axes[0].imshow(image)
axes[1].imshow(sq_open); axes[1].set_title('Open')
axes[2].imshow(sq_open_close); axes[2].set_title('Open then Close')   


# <markdowncell>
# ## Summary
# 
# To summarize visually all operations, we can show the differences:


# <codecell>

from skimage import img_as_float
from matplotlib.colors import LinearSegmentedColormap
def show_diff(im1, im2, axes=None):
    if (axes is None):
        axes = plt
    fim1=img_as_float(im1)
    fim2=img_as_float(im2)
    diffcm=LinearSegmentedColormap.from_list('diffcm', [[0,0,0],[0.25,1.0,0],[0.5,0,0],[1,1,1]], N=4)
    axes.imshow(0.5*fim1+0.25*fim2,cmap=diffcm,vmin=0,vmax=1)

fig, axes = plt.subplots(2,3, figsize=(12,6))
axes[0,0].imshow(image)
show_diff(image, sq_erode, axes=axes[0,1]); axes[0,1].set_title('Erode')
show_diff(image, sq_open, axes=axes[0,2]); axes[0,2].set_title('Open')
show_diff(image, sq_dilate, axes=axes[1,1]); axes[1,1].set_title('Dilate')
show_diff(image, sq_close, axes=axes[1,2]); axes[1,2].set_title('Close')
show_diff(image, sq_open_close, axes=axes[1,0]); axes[1,0].set_title('Open and Close')   


# <markdowncell>
# ### Exercise
# Use morphological operations to clean a segmentation


# <codecell>

plt.rcParams['image.cmap']='gray'   


# <codecell>

from skimage import data, color
hub = color.rgb2gray(data.hubble_deep_field()[350:450, 90:190])
plt.imshow(hub, vmin=0, vmax=1);
skdemo.colorbars()

sq = morphology.square(width=5)

fig=plt.figure()
plt.imshow(hub>0.4)

fig=plt.figure()
plt.imshow(morphology.opening(hub>0.4,sq))

fig=plt.figure()
plt.imshow(hub * morphology.opening(hub>0.4,sq))

fig=plt.figure()
plt.imshow(hub * morphology.dilation(morphology.opening(hub>0.4,sq),sq))

fig=plt.figure()
plt.imshow(morphology.dilation(morphology.opening(hub,sq),sq))   


# <markdowncell>
# Question: Remove the smaller objects to retrieve the large galaxy.


# <codecell>

# your code here   
