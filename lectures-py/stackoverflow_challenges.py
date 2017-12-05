
# coding: utf-8



# <codecell>

from __future__ import division, print_function
%matplotlib inline   


# <codecell>

import matplotlib.pyplot as plt
import numpy as np
from skimage import (filter as filters, io, color,
                     exposure, segmentation, morphology, img_as_float)   


# <markdowncell>
# # Snakes
# 
# Based on http://stackoverflow.com/questions/8686926/python-image-processing-
# help-needed-for-corner-detection-in-preferably-pil-or/9173430#9173430
# 
# <img src="../images/snakes.png" width="200px" style="float: left; padding-
# right: 1em;"/>
# 
# Consider the zig-zaggy snakes on the left (``../images/snakes.png``).  Write
# some code to find the begin- and end-points of each.
# 
# <div style="clear: both;"></div>
# 
# *Hints:*
# 
# 1. Binarize and skeletonize (``morphology.skeletonize``)
# 2. Locate corners via convolution (``scipy.signal.convolve2d``)
# 3. Find intersections between corners and snakes (``np.logical_and``)


# <codecell>

from scipy.signal import convolve2d

img = color.rgb2gray(io.imread('../images/snakes.png'))

# Reduce all lines to one pixel thickness
snakes = morphology.skeletonize(img < 1)

# Find pixels with only one neighbor
corners = convolve2d(snakes, [[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], mode='same') == 1
corners = corners & snakes

# Those are the start and end positions of the segments
y, x = np.where(corners)

plt.figure(figsize=(10, 5))
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
plt.scatter(x, y)
plt.axis('off')
plt.show()   


# <markdowncell>
# # Parameters of a pill
# 
# (Based on StackOverflow http://stackoverflow.com/questions/28281742/fitting-a-
# circle-to-a-binary-image)
# 
# <img src="../images/round_pill.jpg" width="200px" style="float: left; padding-
# right: 1em;"/>
# Consider a pill from the [NLM Pill Image Recognition
# Pilot](http://pir.nlm.nih.gov/pilot/instructions.html)
# (``../images/round_pill.jpg``).  Fit a circle to the pill outline and compute
# its area.
# 
# <div style="clear: both;"></div>
# 
# *Hints:*
# 
# 1. Equalize (``exposure.equalize_*``)
# 2. Detect edges (``filter.canny`` or ``feature.canny``--depending on your
# version)
# 3. Fit the ``CircleModel`` using ``measure.ransac``.


# <codecell>

image = io.imread("../images/round_pill.jpg")
image_equalized = exposure.equalize_adapthist(image)
edges = filters.canny(color.rgb2gray(image_equalized))

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 8))
ax0.imshow(image)
ax1.imshow(image_equalized)
ax2.imshow(edges, cmap='gray');   


# <codecell>

from skimage import measure

coords = np.column_stack(np.nonzero(edges))

model, inliers = measure.ransac(coords, measure.CircleModel,
                                min_samples=3, residual_threshold=1,
                                max_trials=500)

print('Circle parameters:', model.params)

row, col, radius = model.params

f, ax = plt.subplots()
ax.imshow(image, cmap='gray');
circle = plt.Circle((col, row), radius=radius, edgecolor='green', linewidth=2, fill=False)
ax.add_artist(circle);   


# <markdowncell>
# # Viscous fingers
# 
# Based on StackOverflow: http://stackoverflow.com/questions/23121416/long-
# boundary-detection-in-a-noisy-image
# 
# <img src="../images/fingers.png" width="200px" style="float: left; padding-
# right: 1em;"/>
# 
# Consider the fluid experiment on the right.  Determine any kind of meaningful
# boundary.
# 
# <div style="clear: both;"></div>
# 
# *Hints:*
# 
# 1. Convert to grayscale
# 2. Try edge detection (``filters.canny``)
# 3. If edge detection fails, denoising is needed (try
# ``restoration.denoise_tv_bregman``)
# 4. Try edge detection (``filters.canny``)


# <codecell>

from skimage import restoration, color, io, filter as filters, morphology

image = color.rgb2gray(io.imread('../images/fingers.png'))
denoised = restoration.denoise_tv_bregman(image, 1)
edges = filters.canny(denoised, low_threshold=0.01, high_threshold=0.21)

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(denoised, cmap='gray')
axes[1].imshow(edges, cmap='gray')
for ax in axes:
    ax.set_axis_off()   


# <markdowncell>
# # Counting coins
# 
# Based on StackOverflow http://stackoverflow.com/questions/28242274/count-
# number-of-objects-using-watershed-algorithm-scikit-image
# 
# Consider the coins image from the scikit-image example dataset, shown below.
# Write a function to count the number of coins.
# 
# The procedure outlined here is a bit simpler than in the notebook lecture (and
# works just fine!)
# 
# <div style="clear: both;"></div>
# 
# *Hint:*
# 
# 1. Equalize
# 2. Threshold (``filter.otsu`` or ``filters.otsu``, depending on version)
# 3. Remove objects touching boundary (``segmentation.clear_border``)
# 4. Apply morphological closing (``morphology.closing``)
# 5. Remove small objects (``measure.regionprops``)
# 6. Visualize (potentially using ``color.label2rgb``)


# <codecell>

from skimage import data
plt.imshow(data.coins(), cmap='gray');   


# <codecell>

from scipy import ndimage
from skimage import segmentation

image = data.coins()
equalized = exposure.equalize_adapthist(image)
edges = equalized > filters.threshold_otsu(equalized)
edges = segmentation.clear_border(edges)
edges = morphology.closing(edges, morphology.square(3))

f, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(image, cmap='gray')
ax1.imshow(edges, cmap='gray');   


# <codecell>

labels = measure.label(edges)
for region in measure.regionprops(labels):
    if region.area < 200:
        rows, cols = region.coords.T
        labels[rows, cols] = 0

print("Number of coins:", len(np.unique(labels)) - 1)
        
out = color.label2rgb(labels, image, bg_label=0)
plt.imshow(out);   


# <markdowncell>
# # Color wheel
# 
# Based on http://stackoverflow.com/questions/21618252/get-blue-colored-
# contours-using-scikit-image-opencv/21661395#21661395
# 
# <img src="../images/color-wheel.jpg" width="200px" style="float: left;
# padding-right: 1em;"/>
# <img src="../images/balloon.jpg" width="200px" style="float: right; padding-
# left: 1em;"/>
# 
# Consider the color wheel (``../images/color-wheel.jpg``) or the balloon
# (``../images/balloon.jpg``). Isolate all the blue-ish colors in the top
# quadrant.


# <codecell>

from skimage import img_as_float

image = img_as_float(io.imread('../images/color-wheel.jpg'))

blue_lab = color.rgb2lab([[[0, 0, 1.]]])
light_blue_lab = color.rgb2lab([[[0, 1, 1.]]])
red_lab = color.rgb2lab([[[1, 0, 0.]]])
image_lab = color.rgb2lab(image)

distance_blue = color.deltaE_cmc(blue_lab, image_lab, kL=0.5, kC=0.5)
distance_light_blue = color.deltaE_cmc(light_blue_lab, image_lab, kL=0.5, kC=0.5)
distance_red = color.deltaE_cmc(red_lab, image_lab, kL=0.5, kC=0.5)
distance = distance_blue + distance_light_blue - distance_red
distance = exposure.rescale_intensity(distance)

image_blue = image.copy()
image_blue[distance > 0.3] = 0

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))
ax0.imshow(image)
ax1.imshow(distance, cmap='gray')
ax2.imshow(image_blue)
plt.show()   


# <markdowncell>
# # Hand-coin
# 
# Based on StackOverflow http://stackoverflow.com/questions/27910187/how-do-i-
# calculate-the-measurements-of-a-hand-using-scikit-image
# 
# <img src="../images/hand-coin.jpg" width="200px" style="float: left; padding-
# right: 1em;"/>
# 
# Consider the image of the hand and the coin (``../images/hand-coin.jpg``).
# Roughly isolate the region of the hand and plot its orientation.
# 
# <div style="clear: both;"></div>
# 
# *Hint:*
# 
# 1. Segment the image, using ``segmentation.slic``
# 2. Compute the region properties of the resulting labeled image
# 3. Select the largest and second largest (non-background) region--the hand and
# the coin
# 4. For the hand, use ``region.major_axis_length`` and ``region.orientation``
# (where region
#    is your region property) to plot its orientation


# <codecell>

image = io.imread("../images/hand-coin.jpg")

label_image = segmentation.slic(image, n_segments=2)
label_image = measure.label(label_image)

regions = measure.regionprops(label_image)
areas = [r.area for r in regions]
ix = np.argsort(areas)

hand = regions[ix[-1]]
coin = regions[ix[-2]]

selected_labels = np.zeros_like(image[..., 0], dtype=np.uint8)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))

for n, region in enumerate([hand, coin]):
    selected_labels[region.coords[:, 0], region.coords[:, 1]] = n + 2

    y0, x0 = region.centroid
    orientation = region.orientation

    x1 = x0 + np.cos(orientation) * 0.5 * region.major_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * region.major_axis_length
    x2 = x0 - np.sin(orientation) * 0.5 * region.minor_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * region.minor_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

image_label_overlay = color.label2rgb(selected_labels, image=image, bg_label=0)
ax.imshow(image_label_overlay, cmap='gray')
ax.axis('image')
plt.show()   


# <markdowncell>
# ---
# 
# <div style="height: 400px;"></div>


# <codecell>

%reload_ext load_style
%load_style ../themes/tutorial.css   
