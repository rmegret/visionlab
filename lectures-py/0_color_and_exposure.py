
# coding: utf-8



# <markdowncell>
# <slide>
# # Color and exposure
# 
# __Outline:__
# - Basic image manipulation
# - Histograms
# - Color spaces


# <markdowncell>
# <->
# As discussed earlier, images are just numpy arrays. The numbers in those
# arrays correspond to the intensity of each pixel (or, in the case of a color
# image, the intensity of a specific color). To manipulate these, `scikit-image`
# provides the `color` and `exposure` modules.


# <codecell>

%load_ext autoreload
%autoreload 2   


# <codecell>

# Get rid of some annoying warnings (use "always" if looking for weird bugs)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Basic imports for Numpy and Matplotlib
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from skimage import data
import skdemo   


# <markdowncell>
# <slide>
# ## Intro


# <markdowncell>
# <->
# Recall that color images are arrays with pixel rows and columns as the first
# two dimensions (just like a gray-scale image), plus a 3rd dimension that
# describes the RGB color channels.


# <codecell>

from skimage import img_as_float

color_image = data.chelsea()
color_image = img_as_float(color_image)

print(color_image.shape) 
plt.imshow(color_image);   


# <codecell>

from skimage import color
lab = color.rgb2lab(color_image)
#skdemo.imshow_all(lab[...,0],lab[...,1],lab[...,2])

#plt.figure()
#plt.imshow(lab[...,1] > 20)
skdemo.imshow_all(lab[...,0],lab[...,1],lab[...,2])
skdemo.colorbars()

mask=(lab[...,1]<2) | (lab[...,1]>30)
skdemo.imshow_all(lab[...,1]<2, lab[...,1]>30,
                  mask,
                  titles=['eyes','nose', 'both'])
   


# <codecell>

image2 = color_image.copy()
#image2[mask,:]=[255,0,0]
#plt.imshow(image2)

mask1=np.atleast_3d(lab[...,1]<3)
mask2=np.atleast_3d(lab[...,1]>30)

print(color_image.shape, np.atleast_3d((~mask).astype(float)).shape)
fmask = np.atleast_3d((~mask).astype(float)) * 0.5 + 0.5
image2 = color_image * (~mask1 & ~mask2) + mask1 * [1,0,0] + mask2 * [0,0,1]
plt.imshow(image2)

#fig = plt.figure()
#plt.imshow((~mask).astype(float))
#skdemo.colorbars()   


# <codecell>

## fig, axes = plt.subplots(1,3,figsize=(10,3))
axes[0].hist(lab[...,0].ravel(),100);
axes[1].hist(lab[...,1].ravel(),100);
axes[2].hist(lab[...,2].ravel(),100);   


# <markdowncell>
# <slide>
# ## Basic image manipulation


# <markdowncell>
# <subslide>
# ### Slicing and indexing


# <markdowncell>
# <->
# Since images are just arrays, we can manipulate them as we would any other
# array.
# 
# Let's say we want to plot just the red channel of the color image above. We
# know that the red channel is the first channel of the 3rd image-dimension.
# Since Python is zero-indexed, we can write the following:


# <codecell>

red_channel = color_image[:, :, 0]  # or color_image[..., 0]   


# <markdowncell>
# <subslide>
# But when we plot the red channel...


# <codecell>

plt.imshow(red_channel, cmap='gray');   


# <markdowncell>
# <fragment>
# Obviously that's not red at all. The reason is that there's nothing to tell us
# that this array is supposed to be red: It's just a 2D array with a height,
# width, and intensity value---and no color information.
# 
# The green channel is usually closest to the grey-scale version of the image.
# Note that converting to grayscale cannot simply be done by taking the mean of
# the three channels, since the eye is more sensitive to green than to red than
# to blue.  For that purpose, use ``skimage.color.rgb2gray``, which weighs each
# channel appropriately.


# <codecell>

red_channel.shape   


# <markdowncell>
# <skip>
# ---


# <markdowncell>
# <slide>
# ### <span class="exercize">Exercise: three colours</span>


# <markdowncell>
# <->
# Consider the following image:


# <codecell>

from skimage import io
color_image = io.imread('../images/balloon.jpg')
plt.imshow(color_image);   


# <markdowncell>
# <subslide>
# Split this image up into its three components, red, green and blue, and
# display each separately.
# 
# HINT: To display multiple images, we provide you with a small utility library
# called ``skdemo``:
# 
# ```python
# import skdemo
# skdemo.imshow_all(image0, image1, image2, ...)
# ```


# <codecell>

import skdemo

# Uncomment and press <TAB> to see available functions in `skdemo`
#skdemo.imshow_all()   


# <codecell>

# This code is just a template to get you started.
red_image = color_image[:,:,0] # TODO: replace with your code
green_image = color_image[:,:,1] # TODO
blue_image = color_image[:,:,2] # TODO

skdemo.imshow_all(color_image, red_image, green_image, blue_image,
                 titles=['input', 'R', 'G', 'B'])
f=plt.gcf()
plt.sca(f.axes[1])
f.axes[1].set_xticks(range(0,200,32))
skdemo.colorbars(f.axes[1])   


# <markdowncell>
# <subslide>
# Or, simply use ``matplotlib.subplots``:


# <codecell>

fig, axes = plt.subplots(2, 2, figsize=(6,4))
axes=axes.ravel()

axes[0].imshow(color_image)
axes[0].set_title('color')

plt.sca(axes[1])
plt.imshow(red_image)
plt.title('R')
plt.xticks(range(0,200,32))

plt.sca(axes[2])
plt.imshow(green_image)
plt.title('G')

plt.sca(axes[3])
plt.imshow(blue_image)
plt.title('R')   


# <markdowncell>
# <skip>
# ---


# <markdowncell>
# <slide>
# ### Combining color-slices with row/column-slices


# <markdowncell>
# <->
# In the examples above, we just manipulate the last axis of the array (i.e. the
# color channel). As with any NumPy array, however, slicing can be applied to
# each axis:


# <codecell>

color_patches = color_image.copy()
# Remove green (1) & blue (2) from top-left corner.
color_patches[:100, :100, 1:] = 0
# Remove red (0) & blue (2) from bottom-right corner.
color_patches[-100:, -100:, (0, 2)] = 0
plt.imshow(color_patches);   


# <markdowncell>
# <slide>
# ## Histograms


# <markdowncell>
# <->
# Histograms are a quick way to get a feel for the global statistics of the
# image intensity. For example, they can tell you where to set a threshold or
# how to adjust the contrast of an image.


# <markdowncell>
# <slide>
# ### Histograms of images


# <markdowncell>
# <notes>
# For this section, we're going to use a custom plotting function that adds a
# few tweaks to pretty-it-up:
# * Plot the image next to the histogram
# * Plot each RGB channel separately
# * Automatically flatten channels
# * Select reasonable bins based on the image's `dtype`


# <codecell>

# Uncomment to see help on `imshow_with_histogram`
#skdemo.imshow_with_histogram?   


# <markdowncell>
# <notes>
# Using this function, let's look at the histogram of a grayscale image:


# <codecell>

image = data.camera()
skdemo.imshow_with_histogram(image);
skdemo.colorbars()   


# <markdowncell>
# <notes>
# An image histogram shows the number of pixels at each intensity value (or
# range of intensity values, if values are binned). Low-intensity values are
# closer to black, and high-intensity values are closer to white.
# 
# Notice that there's a large peak at an intensity of about 10: This peak
# corresponds with the man's nearly black coat. The peak around the middle of
# the histogram is due to the predominantly gray tone of the image.


# <markdowncell>
# <notes>
# Now let's look at our color image:


# <codecell>

cat = data.chelsea()
skdemo.imshow_with_histogram(cat, xlim=[0,255]);   


# <markdowncell>
# ### Compute and plot histograms using core packages


# <markdowncell>
# How is it done with core functions ?
# We might be inclined to just passs the image to the `plt.hist` function which
# plots histograms:


# <codecell>

plt.hist(image.ravel(),range(0,255));   


# <markdowncell>
# <notes>
# That didn't work as expected. How would you fix the call above to make it work
# correctly?
# (Hint: that's a 2-D array, we need first to flatten it using ``numpy.ravel``)


# <codecell>

# Uncomment to see help on the following functions
#np.ravel?
#plt.hist?   


# <codecell>

image = cat[...,1]  # Extract channel G
print("image.shape           =",image.shape, "  image.size =",image.size)
print("np.ravel(image).shape =",np.ravel(image).shape)
vals, bins, patches = plt.hist(np.ravel(image), color='g')   


# <codecell>

plt.plot((bins[0:-1]+bins[1:])/2, vals, '.-')   


# <markdowncell>
# But this just plots, and does not give us access to the histogram values
# directly to plot it in a different way


# <markdowncell>
# For more flexibility in the plotting, need to do in 2 steps:
# - first compute the values of the histogram (`exposure.histogram`)
# - then plot these values using `plt.plot` or `plt.bar`:


# <codecell>

from skimage import exposure

# Uncomment to see help for `histogram`
#exposure.histogram?   


# <codecell>

hist, bin_centers = exposure.histogram(image)

fig, axes = plt.subplots(1,2, figsize=(12,3))
axes[0].plot(bin_centers, hist, 'g')
bin_step=bin_centers[1]-bin_centers[0]
axes[1].bar(bin_centers, hist, width=bin_step, color='g')   


# <markdowncell>
# <notes>
# As you can see, the intensity for each RGB channel is plotted separately.
# Unlike the previous histogram, these histograms almost look like Gaussian
# distributions that are shifted. This reflects the fact that intensity changes
# are relatively gradual in this picture: There aren't very many uniform
# instensity regions in this image.


# <markdowncell>
# <notes>
# **Note:** While RGB histograms are pretty, they are often not very intuitive
# or useful, since a high red value is very different when combined with *low*
# green/blue values (the result will tend toward red) vs *high* green and blue
# values (the result will tend toward white).


# <markdowncell>
# <slide>
# ### Histograms and contrast


# <markdowncell>
# <notes>
# Enhancing the contrast of an image allow us to more easily identify features
# in an image, both by eye and by detection algorithms.
# 
# Let's take another look at the gray-scale image from earlier:


# <codecell>

from skimage import img_as_float, img_as_ubyte
image = data.camera()
axi,axh = skdemo.imshow_with_histogram(image);   


# <markdowncell>
# <notes>
# Notice the intensity values at the bottom. Since the image has a `dtype` of
# `uint8`, the values go from 0 to 255. Though you can see some pixels tail off
# toward 255, you can clearly see in the histogram, and in the image, that we're
# not using the high-intensity limits very well.
# 
# Based on the histogram values, you might want to take all the pixels values
# that are more than about 180 in the image, and make them pure white (i.e. an
# intensity of 255). While we're at it, values less than about 10 can be set to
# pure black (i.e. 0). We can do this easily using `rescale_intensity`, from the
# `exposure` subpackage.


# <codecell>

from skimage import exposure
high_contrast = exposure.rescale_intensity(image, in_range=(160, 170), out_range=(150,200))

skdemo.imshow_with_histogram(image.astype(np.uint8),xlim=[-5,260]);
skdemo.colorbars()
skdemo.imshow_with_histogram(high_contrast,xlim=[-5,260]);
skdemo.colorbars()
fig = plt.figure()
skdemo.imshow_all(image,high_contrast)   


# <markdowncell>
# <notes>
# The contrast is visibly higher in the image, and the histogram is noticeably
# stretched. The sharp peak on the right is due to all the pixels greater than
# 180 (in the original image) that were piled into a single bin (i.e. 255).


# <codecell>

# Uncomment to see help on `rescale_intensity`
#exposure.rescale_intensity?   


# <markdowncell>
# <notes>
# Parameters for `rescale_intensity`:
# * `in_range`: The min and max intensity values desired in the input image.
# Values below/above these limits will be clipped.
# * `out_range`: The min and max intensity values of the output image. Pixels
# matching the limits from `in_range` will be rescaled to these limits.
# Everything in between gets linearly interpolated.


# <markdowncell>
# <slide>
# ### Histogram equalization


# <markdowncell>
# <notes>
# In the previous example, the grayscale values (10, 180) were set to (0, 255),
# and everything in between was linearly interpolated. There are other
# strategies for contrast enhancement that try to be a bit more intelligent---
# notably histogram equalization.
# 
# Let's first look at the cumulative distribution function (CDF) of the image
# intensities.


# <codecell>

ax_image, ax_hist = skdemo.imshow_with_histogram(image)
skdemo.plot_cdf(image, ax=ax_hist.twinx())   


# <markdowncell>
# <notes>
# For each intensity value, the CDF gives the fraction of pixels *below* that
# intensity value.
# 
# One measure of contrast is how evenly distributed intensity values are: The
# dark coat might contrast sharply with the background, but the tight
# distribution of pixels in the dark coat mean that details in the coat are
# hidden.


# <markdowncell>
# <fragment>
# To enhance contrast, we could *spread out intensities* that are tightly
# distributed and *combine intensities* which are used by only a few pixels.


# <markdowncell>
# <notes>
# This redistribution is exactly what histogram equalization does. And the CDF
# is important because a perfectly uniform distribution gives a CDF that's a
# straight line. We can use `equalize_hist` from the `exposure` package to
# produce an equalized image:


# <codecell>

equalized = exposure.equalize_hist(image)   


# <codecell>

ax_image, ax_hist = skdemo.imshow_with_histogram(equalized)
skdemo.plot_cdf(equalized, ax=ax_hist.twinx())   


# <markdowncell>
# <notes>
# The tightly distributed dark-pixels in the coat have been spread out, which
# reveals many details in the coat that were missed earlier. As promised, this
# more even distribution produces a CDF that approximates a straight line.
# 
# Notice that the image intensities switch from 0--255 to 0.0--1.0:


# <codecell>

equalized.dtype   


# <markdowncell>
# <notes>
# Functions in `scikit-image` allow any data-type as an input, but the output
# data-type may change depending on the algorithm. While `uint8` is really
# efficient in terms of storage, we'll see in the next section that computations
# using `uint8` images can be problematic in certain cases.
# 
# If you need a specific data-type, check out the image conversion functions in
# scikit image:


# <codecell>

import skimage

# Uncomment and press TAB
#skimage.img_as  # <TAB>   


# <markdowncell>
# <slide>
# ### Contrasted-limited, adaptive histogram equalization


# <markdowncell>
# <notes>
# Unfortunately, histogram equalization tends to give an image whose contrast is
# artificially high. In addition, better enhancement can be achieved locally by
# looking at smaller patches of an image, rather than the whole image. In the
# image above, the contrast in the coat is much improved, but the contrast in
# the grass is somewhat reduced.
# 
# Contrast-limited adaptive histogram equalization (CLAHE) addresses these
# issues. The implementation details aren't too important, but seeing the result
# is helpful:


# <codecell>

equalized = exposure.equalize_adapthist(image)   


# <codecell>

ax_image, ax_hist = skdemo.imshow_with_histogram(equalized)
skdemo.plot_cdf(equalized, ax=ax_hist.twinx())   


# <markdowncell>
# <notes>
# Compared to plain-old histogram equalization, the high contrast in the coat is
# maintained, but the contrast in the grass is also improved.
# Furthermore, the contrast doesn't look overly-enhanced, as it did with the
# standard histogram equalization.
# 
# Again, notice that the output data-type is different than our input. This
# time, we have a `uint16` image, which is another common format for images:


# <codecell>

equalized.dtype   


# <markdowncell>
# <notes>
# There's a bit more tweaking involved in using `equalize_adapthist` than in
# `equalize_hist`: Input parameters are used to control the patch-size and
# contrast enhancement. You can learn more by checking out the docstring:


# <codecell>

# Uncomment to display help on `equalize_adapthist`
#exposure.equalize_adapthist?   


# <markdowncell>
# <slide>
# ### Histograms and thresholding


# <markdowncell>
# <notes>
# One of the most common uses for image histograms is thresholding. Let's return
# to the original image and its histogram


# <codecell>

skdemo.imshow_with_histogram(image);   


# <markdowncell>
# <notes>
# Here the man and the tripod are fairly close to black, and the rest of the
# scene is mostly gray. But if you wanted to separate the two, how do you decide
# on a threshold value just based on the image? Looking at the histogram, it's
# pretty clear that a value of about 50 will separate the two large portions of
# this image.


# <codecell>

threshold = 145
threshold2 = 87

ax_image, ax_hist = skdemo.imshow_with_histogram(image)
# This is a bit of a hack that plots the thresholded image over the original.
# This just allows us to reuse the layout defined in `plot_image_with_histogram`.
#ax_image.imshow( (image > threshold2) )
ax_image.imshow( (image < threshold) & (image > threshold2) )
ax_hist.axvline(threshold, color='red');
ax_hist.axvline(threshold2, color='blue');   


# <markdowncell>
# <notes>
# Note that the histogram plotted here is for the image *before* thresholding.
# 
# This does a pretty good job of separating the man (and tripod) from most of
# the background. Thresholding is the simplest method of image segmentation;
# i.e. dividing an image into "meaningful" regions. More on that later.
# 
# As you might expect, you don't have to look at a histogram to decide what a
# good threshold value is: There are (many) algorithms that can do it for you.
# For historical reasons, `scikit-image` puts these functions in the `filter`
# module.
# 
# One of the most popular thresholding methods is Otsu's method, which gives a
# slightly different threshold than the one we defined above:


# <codecell>

# Rename module so we don't shadow the builtin function
import skimage.filters as filters
threshold = filters.threshold_otsu(image)
print(threshold)   


# <codecell>

skdemo.imshow_with_histogram(image)   


# <codecell>

image = data.coins()
plt.imshow(image > threshold);   


# <markdowncell>
# <notes>
# Note that the features of the man's face are slightly better resolved in this
# case.
# 
# You can find a few other thresholding methods in the `filter` module:


# <codecell>

import skimage.filters as filters
#filters.threshold  # <TAB>   


# <markdowncell>
# <slide>
# ## Color spaces


# <markdowncell>
# <notes>
# While RGB is fairly easy to understand, using it to detect a specific color
# (other than red, green, or blue) can be a pain. Other color spaces often
# devote a single component to the image intensity (a.k.a. luminance, lightness,
# or value) and two components to represent the color (e.g. hue and saturation
# in [HSL and HSV](http://en.wikipedia.org/wiki/HSL_and_HSV)).
# 
# You can easily convert to a different color representation, or "color space",
# using functions in the `color` module.


# <codecell>

from skimage import color
#color.rgb2  # <TAB>   


# <codecell>

plt.imshow(color_image);   


# <markdowncell>
# <notes>
# ### LAB colorspace
# 
# Here, we'll look at the LAB (a.k.a. CIELAB) color space (`L` = luminance, `a`
# and `b` define a 2D description of the actual color or "hue"):


# <codecell>

from skimage import color

color_image = io.imread('../images/balloon.jpg')
lab_image = color.rgb2lab(color_image)
print(lab_image.shape, lab_image.dtype)   


# <markdowncell>
# <notes>
# Converting to LAB didn't change the shape of the image at all. Let's try to
# plot it:


# <codecell>

#plt.imshow(lab_image);   


# <markdowncell>
# Depending on your version on Matplotlib, you may end up with garbage or an
# error:
# 
#     ValueError: Floating point image RGB values must be in the 0..1 range.
# 
# This is because `imshow` assume that 3 channels images passed as arguments are
# RGB. Lab images do not have the same dynamics, which prevent displaying them
# directly that way. Let's look at the L, a, b channels individually:


# <codecell>

from matplotlib.colors import LinearSegmentedColormap
skdemo.imshow_all(color_image, lab_image[..., 0], lab_image[..., 1], lab_image[..., 2],
                 titles=['RGB', 'L', 'a', 'b'], limits='auto', size=4)

# Tweak a bit the display to highlight what information each channel brings
plt.gcf().axes[1].get_images()[0].set_clim(0,100)
#plt.gcf().axes[2].get_images()[0].set_cmap(LinearSegmentedColormap.from_list('GnMg',[[0.0,1,0.0],[0.9,0,0.1]]))
#plt.gcf().axes[3].get_images()[0].set_cmap(LinearSegmentedColormap.from_list('BuYl',[[0,0,1],[1,1,0]]))

# Add colorbars except for RGB
skdemo.colorbars(axes=plt.gcf().axes[1:]); plt.tight_layout()   


# <markdowncell>
# Lab gamut, showing only sRGB colors:
# <img src="figs/Lab_color_space.svg" width="60%"/>
# Image <a
# href="https://commons.wikimedia.org/wiki/File:Lab_color_space.svg">licensed CC
# BY-SA by Jacob Rus</a> (modified for horizontal layout)
# 
# `Lab` colorspace has two advantages:
# - it separates luminance `L` from chrominance information `a` and `b`;
# - it is more perceptually uniform than sRGB to better approximates human
# vision.


# <markdowncell>
# <->
# ---


# <markdowncell>
# ## <span class="exercize">Exercise: Fun with film</span>


# <markdowncell>
# In the film industry, it is often necessary to impose actors on top of a
# rendered background.  To do that, the actors are filmed on a "green screen".
# Here's an example shot (``images/greenscreen.jpg``):
# 
# <img src="../images/greenscreen.jpg" width="300px"/>
# 
# Say we'd like to help these friendly folk travel into a rainforest
# (``images/forest.jpg``):
# 
# <img src="../images/forest.jpg" width="300px"/>
# 
# Can you help them by transplanting the foreground of the greenscreen onto the
# backdrop of the forest?


# <codecell>

from skimage import img_as_float, img_as_ubyte
from skimage.transform import resize
image = io.imread('../images/greenscreen.jpg')
bg = io.imread('../images/forest.jpg')
image = img_as_float(image)
bg = img_as_float(bg)

bg = resize(bg, image.shape)  # need to resize background to same size as image

skdemo.imshow_with_histogram(img_as_ubyte(image), xlim=[0,255]);

mask=(image[:,:,1]>150/255) & (image[:,:,0]<100/255)
lab = color.rgb2lab(image)
mask = lab[...,1]<-40

fmask = np.atleast_3d(mask) #mask.reshape( (375,500,1) )

out=image.copy()
out = image*(1-fmask) + bg * fmask
#out[mask,:] = bg[mask,:]

skdemo.imshow_all(image, bg, mask, out)   


# <codecell>

a = np.array(range(7)).reshape(1,7)
b = np.array(range(0,10,2)).reshape(5,1)

print('a=\n',a)

print('b=\n',b)

print('a+b=\n',a+b)   


# <markdowncell>
# <notes>
# Matplotlib expected an RGB array, and apparently, LAB arrays don't look
# anything like RGB arrays.
# 
# That said, there is some resemblance to the original. To get some sense of the
# Lab colorspace,
# let's display instead each channel separately.


# <markdowncell>
# <fragment>
# ## <span class="exercize">Exercise: Exploring color spaces</span>


# <markdowncell>
# 1. Decompose the Balloon image of earlier into its H, S and V (hue,
# saturation, value) color components.  Display each component and interpret the
# result.


# <codecell>

# Your code here   


# <markdowncell>
# <fragment>
# 2. Use the LAB color space to **isolate the eyes** in the `chelsea` image.
# Plot the L, a, b components to get a feel for what they do, and see the
# [wikipedia page](http://en.wikipedia.org/wiki/Lab_colorspace) for more info.
# Bonus: **Isolate the nose** too.


# <codecell>

# Your code here   


# <markdowncell>
# <->
# ---


# <markdowncell>
# <slide>
# ## Further reading


# <markdowncell>
# <->
# * [Example of tinting gray-scale images](<http://scikit-
# image.org/docs/dev/auto_examples/plot_tinting_grayscale_images.html>)
# * Color spaces (see [`skimage.color`](http://scikit-
# image.org/docs/dev/api/skimage.color.html) package)
#   - `rgb2hsv`
#   - `rgb2luv`
#   - `rgb2xyz`
#   - `rgb2lab`
# * [Histogram equalization](http://scikit-
# image.org/docs/dev/auto_examples/plot_equalize.html) and [local histogram
# equalization](http://scikit-
# image.org/docs/dev/auto_examples/plot_local_equalize.html)
